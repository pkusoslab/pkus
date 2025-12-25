import queue
import threading
import sys, os, json, time, mmap, torch, argparse, random, numpy
from pathlib import Path
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file, save_file
from pdb import set_trace as st

import ivshmem_comm as ic
from pkus.model_def import GuestLoraModel, replace_lora_layers

# set different prompt
PROMPT_DATABASE = "sst2"
# PROMPT_DATABASE = "squad"
# PROMPT_DATABASE = "mnli"

# use pruned lora model or not
PRUNED = False

# set different base model
BASE_MODEL = "llama-3-1b"
# BASE_MODEL = "gpt2-large"

# set different lora model (trained from different datasets)
LORA_DATABASE = "sst2"
# LORA_DATABASE = "squad"
# LORA_DATABASE = "mnli"

# use GPU/CPU if set False
TEST_OUR_METHOD = True

PRUNE_RATIO = 0.8 if LORA_DATABASE == "sst2" else (0.64 if LORA_DATABASE == "squad" else 0.66)


base_model_dir = f"./model/{BASE_MODEL}"
lora_model_dir = f"./model/{BASE_MODEL}-lora{f"/dynamic/ratio={PRUNE_RATIO}" if PRUNED else ""}/{LORA_DATABASE}"

print(lora_model_dir)

sst2_prompt = "hide new secretions from the parental units"
mnli_prompt = ""
squad_prompt = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nQuestion: What sits on top of the Main Building at Notre Dame?"
prompt = sst2_prompt if PROMPT_DATABASE == "sst2" else (squad_prompt if PROMPT_DATABASE == "squad" else mnli_prompt)

# IVSHMEM path on host machine and TEE (guest)
HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

RR = 0.0001 # round-robin sleep interval
DEFAULT_DTYPE = torch.float32




shared_state = {
    'request_id': 0,
    'module_name': None,
    'input_tensor': None,
    'exit_signal': False
}

def guest_worker(worker_id, guest_lora_model, condition, output_queue):
    global shared_state
    guest_forward_time = 0.0    
    last_processed_id = 0
    # if worker_id == 0:
    #     log_f = open("log_guest_worker_0.txt", "a")
    
    while True:
        module_name, input_tensor = None, None
        
        with condition:
            while shared_state['request_id'] == last_processed_id and not shared_state['exit_signal']:
                condition.wait()
            
            if shared_state['exit_signal']:
                print(f"Worker {worker_id} - forward时间累计 {guest_forward_time:.6f} 秒")
                break
            
            last_processed_id = shared_state['request_id']
            module_name = shared_state['module_name']
            input_tensor = shared_state['input_tensor']
        
        guest_forward_start = time.time()
        # if worker_id == 0:
        #     log_f.write(f"{guest_forward_start}\n")
        delta_h = guest_lora_model.forward(module_name, input_tensor)
        guest_forward_end = time.time()
        # if worker_id == 0:
        #     log_f.write(f"{guest_forward_end}\n")
        output_queue.put((worker_id, delta_h))

        guest_forward_time += guest_forward_end - guest_forward_start


def guest_main(set_multi_thread=False):
    # open("log_guest.txt", "w").close()
    # open("log_tensor2bytes.txt", "w").close()
    # open("log_tensor_bytes_and_module_name2blocks.txt", "w").close()
    # open("log_blocks2tensor_bytes_and_module_name.txt", "w").close()
    # open("log_bytes2tensor.txt", "w").close()
    # open("log_guest_master.txt", "w").close()
    # open("log_guest_worker_0.txt", "w").close()
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        ic.write_host_guest_uint8(shm, 1)
        while True:
            blocks = ic.read_blocks(shm, "guest")
            if len(blocks) > 0:
                split_point = ic.get_msg_id(blocks[0])
                lora_weight_blocks = blocks[:split_point]
                lora_config_blocks = blocks[split_point:]
                break
            else:
                time.sleep(RR)

        serialized_weights = ic.blocks2bytes(lora_weight_blocks)
        serialized_config = ic.blocks2bytes(lora_config_blocks)
        print(f"LoRA weights size: {len(serialized_weights) / 1024:.2f} KB")
        print(f"LoRA config size: {len(serialized_config) / 1024:.2f} KB")
        lora_state_dict, lora_config_dict = ic.bytes2lora_weight_config(serialized_weights, serialized_config)
        guest_lora_model = GuestLoraModel(lora_state_dict, lora_config_dict)
        device = torch.device("cpu")
        guest_lora_model._modules.to(device)
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 1) # notify host that guest is ready
        print("[GUEST] LoRA layers initialized. Waiting for requests...")

        # single thread mode
        if not set_multi_thread:
            all_guest_forward_time = 0.0
            all_host_guest_comm_time = 0.0
            try:
                while True:
                    guest_read_start = time.time()
                    request_blocks = ic.read_blocks(shm, "guest")
                    if len(request_blocks) > 0:
                        guest_read_end = time.time()
                        tensor_bytes, module_name = ic.blocks2tensor_bytes_and_module_name(request_blocks)
                        input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)

                        guest_forward_start = time.time()
                        delta_h = guest_lora_model.forward(module_name, input_tensor)
                        guest_forward_end = time.time()
                        all_guest_forward_time += (guest_forward_end - guest_forward_start)
                        response_bytes = ic.tensor2bytes(delta_h)
                        # msg_id = int(time.time())
                        response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
                        guest_write_start = time.time()
                        ic.write_blocks(shm, response_blocks, "guest")
                        guest_write_end = time.time()
                        all_host_guest_comm_time += (guest_write_end - guest_write_start + guest_read_end - guest_read_start)
                        # with open("log_guest.txt", "a") as log_f:
                        #     log_f.write(f"{(guest_write_end - guest_write_start + guest_read_end - guest_read_start):6f} {(guest_forward_end - guest_forward_start):6f} {(guest_write_start - guest_read_end):6f} {guest_read_start} {guest_write_end}\n")
                    else:
                        time.sleep(RR)
            except KeyboardInterrupt:
                print(f"guest端LoRA前向计算总时间: {all_guest_forward_time:.6f} 秒")
                print(f"guest读写共享内存总时间: {all_host_guest_comm_time:.6f} 秒")
                print(f"{all_host_guest_comm_time:.2f}, {all_guest_forward_time:.2f}")
        # multi thread mode
        else:
            thread_num = 32

            global shared_state
            condition = threading.Condition()
            output_queue = queue.Queue()

            workers = []
            for i in range(thread_num):
                t = threading.Thread(target=guest_worker, 
                                    args=(i, guest_lora_model, condition, output_queue))
                t.daemon = True
                t.start()
                workers.append(t)
                
            total_broadcast_overhead = 0.0 # overhead of main thread writing shared memory and notifing workers
            total_request_latency = 0.0    # interval between the beginning of the request and the reception of the first result

            log_f = open("log_guest_master.txt", "a")
            try:
                while True:
                    request_blocks = ic.read_blocks(shm, "guest")
                    if len(request_blocks) > 0:
                        tensor_bytes, module_name = ic.blocks2tensor_bytes_and_module_name(request_blocks)
                        input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)

                        request_start_time = time.time()
                        with condition:
                            shared_state['request_id'] += 1 
                            shared_state['module_name'] = module_name
                            shared_state['input_tensor'] = input_tensor
                            condition.notify_all()
                        call_workers_end_time = time.time()
                        # log_f.write(f"{call_workers_end_time}\n")
                        total_broadcast_overhead += call_workers_end_time - request_start_time
                        
                        results = []
                        for _ in range(thread_num):
                            results.append(output_queue.get())
                        request_end_time = time.time()
                        # log_f.write(f"{request_end_time}\n")

                        total_request_latency += (request_end_time - request_start_time)
                        
                        worker_id, delta_h = results[0]
                        
                        response_bytes = ic.tensor2bytes(delta_h)
                        response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
                        ic.write_blocks(shm, response_blocks, "guest")
                    else:
                        time.sleep(RR)
                    
            except KeyboardInterrupt:
                print(f"{total_broadcast_overhead:.2f}, {total_request_latency:.2f}")
                
                with condition:
                    shared_state['exit_signal'] = True
                    condition.notify_all()
                    
                for t in workers:
                    t.join()
                
                log_f.close()

def check_lora_weights_zero(adapter_path):
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            return False, []
    
    try:
        if adapter_weights_path.endswith('.safetensors'):
            state_dict = load_file(adapter_weights_path)
        else:
            state_dict = torch.load(adapter_weights_path, map_location='cpu')
    except:
        return False, []
    
    unzero_modules = []
    
    for key in state_dict.keys():
        if 'lora' in key.lower():
            # print(key)
            #st()
            weight = state_dict[key]
            # print(key,weight,torch.allclose(weight, torch.zeros_like(weight), atol=1e-4))
            # st()
            if not torch.allclose(weight, torch.zeros_like(weight), atol=1e-4):
                if "transformer" in key and "llama" in BASE_MODEL.lower():
                    parts = key.split('.')
                    new_key = '.'.join(parts[3:-2])
                    if new_key not in unzero_modules:
                        unzero_modules.append(new_key)
                else:
                    parts = key.split('.')
                    new_key = '.'.join(parts[2:-2])
                    if new_key not in unzero_modules:
                        unzero_modules.append(new_key)

    return len(unzero_modules) > 0, unzero_modules


def check_lora_weights(adapter_path):
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            return False, []
    
    try:
        if adapter_weights_path.endswith('.safetensors'):
            state_dict = load_file(adapter_weights_path)
        else:
            state_dict = torch.load(adapter_weights_path, map_location='cpu')
    except:
        return False, []
    
    lora_modules = []
    
    for key in state_dict.keys():
        if 'lora' in key.lower():
            # print(key)
            #st()
            weight = state_dict[key]
            if "transformer" in key and "llama" in BASE_MODEL.lower():
                parts = key.split('.')
                new_key = '.'.join(parts[3:-2])
                if new_key not in lora_modules:
                    lora_modules.append(new_key)
            else:
                parts = key.split('.')
                new_key = '.'.join(parts[2:-2])
                if new_key not in lora_modules:
                    lora_modules.append(new_key)
    return len(lora_modules) > 0, lora_modules


def host_main():
    open("log_host.txt", "w").close()
    open("log_tensor2bytes.txt", "w").close()
    open("log_tensor_bytes_and_module_name2blocks.txt", "w").close()
    open("log_blocks2tensor_bytes_and_module_name.txt", "w").close()
    open("log_bytes2tensor.txt", "w").close()
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        dtype=DEFAULT_DTYPE,
        device_map=None,
    )
    base_model.to(device)
    
    if PRUNED:
        _, unzero_modules = check_lora_weights_zero(lora_model_dir)
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  
            target_modules=unzero_modules, 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        state_dict = load_file(os.path.join(lora_model_dir, "adapter_model.safetensors"), device="cuda" if torch.cuda.is_available() else "cpu")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                parts = name.split('.')
                if "squad" in PROMPT_DATABASE and "llama" in BASE_MODEL.lower():
                    new_name = '.'.join(parts[3:-3])
                    new_name2 = '.'.join(parts[3:-2])
                else:
                    new_name = '.'.join(parts[2:-3])
                    new_name2 = '.'.join(parts[2:-2])
                # print(new_name, new_name2)
                # st()
                if new_name in unzero_modules:
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]

        # model.load_state_dict(state_dict, strict=False)
    else:
        _, lora_modules = check_lora_weights(lora_model_dir)
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  
            target_modules=lora_modules, 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        state_dict = load_file(os.path.join(lora_model_dir, "adapter_model.safetensors"), device="cuda" if torch.cuda.is_available() else "cpu")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                parts = name.split('.')
                if "squad" in PROMPT_DATABASE and "llama" in BASE_MODEL.lower():
                    new_name = '.'.join(parts[3:-3])
                    new_name2 = '.'.join(parts[3:-2])
                else:
                    new_name = '.'.join(parts[2:-3])
                    new_name2 = '.'.join(parts[2:-2])
                if new_name in lora_modules:
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]
        # model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    total_time = 0.0 # host
    init_time = 0.0 # host
    init_start = time.time()
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

        lora_weights_bytes, lora_config_bytes = ic.lora_weight_config2bytes(model)
        lora_weights_block_num = (len(lora_weights_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_weight_blocks = ic.bytes2blocks(lora_weights_bytes, msg_id=lora_weights_block_num, force_is_not_last=True)
        lora_config_blocks = ic.bytes2blocks(lora_config_bytes, msg_id=lora_weights_block_num)

        packed_blocks = lora_weight_blocks + lora_config_blocks
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        ic.write_blocks(shm, packed_blocks, "host")

        model = replace_lora_layers(model, shm)
    
    while True:
        if ic.read_ret_uint8(shm) == 1:
            break
        else:
            time.sleep(0.01)
            
    init_end = time.time()
    init_time = init_end - init_start
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 禁用采样 -> 更确定性
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    total_end = time.time()
    total_time = total_end - init_start
    # print(f"[HOST] initialization time: {init_time:.6f} s")
    print(f"[HOST] generation time: {gen_end - gen_start:.6f} s")
    print(f"[HOST] length of the output tokens: {len(output_ids[0])}")
    # print(f"[HOST] full time: {total_time:.6f} s")

def test_host():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=DEFAULT_DTYPE,
        device_map=None,
    )
    if PRUNED:
        _, unzero_modules = check_lora_weights_zero(lora_model_dir)
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  
            target_modules=unzero_modules, 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        state_dict = load_file(os.path.join(lora_model_dir, "adapter_model.safetensors"), device="cuda" if torch.cuda.is_available() else "cpu")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                parts = name.split('.')
                if "squad" in PROMPT_DATABASE and "llama" in BASE_MODEL.lower():
                    new_name = '.'.join(parts[3:-3])
                    new_name2 = '.'.join(parts[3:-2])
                else:
                    new_name = '.'.join(parts[2:-3])
                    new_name2 = '.'.join(parts[2:-2])
                # print(new_name, new_name2)
                # st()
                if new_name in unzero_modules:
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]

        # model.load_state_dict(state_dict, strict=False)
    else:
        _, lora_modules = check_lora_weights(lora_model_dir)
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  
            target_modules=lora_modules, 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        state_dict = load_file(os.path.join(lora_model_dir, "adapter_model.safetensors"), device="cuda" if torch.cuda.is_available() else "cpu")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                parts = name.split('.')
                if "squad" in PROMPT_DATABASE and "llama" in BASE_MODEL.lower():
                    new_name = '.'.join(parts[3:-3])
                    new_name2 = '.'.join(parts[3:-2])
                else:
                    new_name = '.'.join(parts[2:-3])
                    new_name2 = '.'.join(parts[2:-2])
                if new_name in lora_modules:
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]
        # model.load_state_dict(state_dict, strict=False)

    model.to(device=device, dtype=DEFAULT_DTYPE)
    model.eval()

    is_peft = isinstance(model, PeftModel)
    print(f"is PeftModel: {is_peft}")
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        LoraLinear = None
    lora_count = sum(1 for _, m in model.named_modules() if LoraLinear is not None and isinstance(m, LoraLinear))
    print(f"LoRA Linear layer count: {lora_count}")

    device_counts = {}
    for name, p in model.named_parameters():
        d = str(p.device)
        device_counts[d] = device_counts.get(d, 0) + 1
    print("Parameter device distribution (device: param_count):", device_counts)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    print(f"[HOST-ONLY] inference time: {gen_end - gen_start:.6f} s")
    print(f"[HOST-ONLY] length of the output tokens: {len(output_ids[0])}")


test_bytes = b"test" * 1048576
test_blocks = ic.bytes2blocks(test_bytes, msg_id=1)
def test_rw_host():
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        for _ in range(1000):
            ic.write_blocks(shm, test_blocks, "host")
            while True:
                read_blocks = ic.read_blocks(shm, "host")
                if len(read_blocks) > 0:
                    break
                time.sleep(RR)
            read_bytes = ic.blocks2bytes(read_blocks)
            try:
                assert read_bytes == test_bytes
            except AssertionError:
                print(f"len of read_bytes: {len(read_bytes)}")

def test_rw_guest():
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        while True:
            while True:
                read_blocks = ic.read_blocks(shm, "guest")
                if len(read_blocks) > 0:
                    break
                time.sleep(RR)
            read_bytes = ic.blocks2bytes(read_blocks)
            try:
                assert read_bytes == test_bytes
            except AssertionError:
                print(f"len of read_bytes: {len(read_bytes)}")
            ic.write_blocks(shm, test_blocks, "guest")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Model Inference")
    parser.add_argument("--role", choices=["host", "guest"], help="Role of the client (host or guest)")
    args = parser.parse_args()
    client_role = args.role
    if client_role == "host":
        if TEST_OUR_METHOD:
            host_main()
        else:
            test_host()
        # test_rw_host()
    else:
        if not TEST_OUR_METHOD:
            test_host()
        else:
            guest_main(set_multi_thread=False)
            # test_rw_guest()