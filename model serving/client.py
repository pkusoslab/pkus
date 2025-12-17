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

# 不同prompt
PROMPT_DATABASE = "sst2"
# PROMPT_DATABASE = "squad"
# PROMPT_DATABASE = "mnli"

# lora剪枝与否
PRUNED = False

# 不同base model
BASE_MODEL = "llama-3-1b"
# BASE_MODEL = "gpt2-large"

# 不同lora模型
LORA_DATABASE = "sst2"
# LORA_DATABASE = "squad"
# LORA_DATABASE = "mnli"

# 我们的方法/纯GPU|CPU
TEST_OUR_METHOD = True

# 剪枝率：0.8和sst2绑定，0.64和squad绑定，0.66和mnli绑定
PRUNE_RATIO = 0.8 if LORA_DATABASE == "sst2" else (0.64 if LORA_DATABASE == "squad" else 0.66)


base_model_dir = f"./model/{BASE_MODEL}"
lora_model_dir = f"./model/{BASE_MODEL}-lora{f"/dynamic/ratio={PRUNE_RATIO}" if PRUNED else ""}/{LORA_DATABASE}"

print(lora_model_dir)

sst2_prompt = "hide new secretions from the parental units" # 是不是应该告知模型这是一个情感分类任务？
mnli_prompt = ""
squad_prompt = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nQuestion: What sits on top of the Main Building at Notre Dame?"
# PROMPT = "How many states does the US have?"
prompt = sst2_prompt if PROMPT_DATABASE == "sst2" else (squad_prompt if PROMPT_DATABASE == "squad" else mnli_prompt)

HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

RR = 0.0001 # 轮询等待时间
DEFAULT_DTYPE = torch.float32

# 根据host构建好模型后发来的lora相关权重和配置，初始化guest端的lora层
class GuestLoraModel:
    def __init__(self, lora_state_dict, lora_config_dict):
        self.lora_config = LoraConfig(**lora_config_dict)
        self._modules = nn.ModuleDict()  # 使用ModuleDict来妥善管理所有层
        # 原始模块名 -> 安全化键（用于 ModuleDict，不含 '.'）
        self._name_map = {}
        self._scaling = self.lora_config.lora_alpha / self.lora_config.r

        # 1. 识别所有独立的LoRA模块
        # 我们支持两种键格式：
        # 1) module...lora_A.weight (无adapter名)
        # 2) module...lora_A.<adapter>.weight  (带adapter名，比如default)
        lora_a_keys = [k for k in lora_state_dict if (".lora_A." in k and k.endswith(".weight")) or k.endswith("lora_A.weight")]

        for a_key in lora_a_keys:
            # 解析两种可能的格式
            if ".lora_A." in a_key and a_key.endswith(".weight"):
                # 带 adapter 名的：module... .lora_A.<adapter>.weight
                module_name, tail = a_key.split(".lora_A.", 1)
                adapter = tail[:-len(".weight")]  # e.g. "default"
                b_key = f"{module_name}.lora_B.{adapter}.weight"
            else:
                # 旧格式或无 adapter：module...lora_A.weight
                module_name = a_key.removesuffix(".lora_A.weight")
                b_key = f"{module_name}.lora_B.weight"

            if b_key not in lora_state_dict:
                print(f"  - Warning: Corresponding LoRA B key not found for '{a_key}', expected '{b_key}'. Skipping.")
                continue

            # 2. 从权重张量的形状推断出维度
            lora_a_weight = lora_state_dict[a_key]
            lora_b_weight = lora_state_dict[b_key]
            if not isinstance(lora_a_weight, torch.Tensor):
                lora_a_weight = torch.tensor(lora_a_weight)
            if not isinstance(lora_b_weight, torch.Tensor):
                lora_b_weight = torch.tensor(lora_b_weight)

            # W_A shape: (rank, in_features)
            # W_B shape: (out_features, rank)
            rank, in_features = lora_a_weight.shape
            out_features, _ = lora_b_weight.shape

            # 3. 创建LoRA A和B线性层（保持与权重形状一致）
            lora_a_layer = nn.Linear(in_features, rank, bias=False)
            lora_b_layer = nn.Linear(rank, out_features, bias=False)

            # 4. 直接拷贝权重到层中
            with torch.no_grad():
                lora_a_layer.weight.data.copy_(lora_a_weight)
                lora_b_layer.weight.data.copy_(lora_b_weight)

            # 5. 将构建好的层存入ModuleDict，ModuleDict不允许键含'.'，将所有'.'替换为'-'
            safe_key = module_name.replace('.', '-')
            self._modules[safe_key] = nn.Sequential(lora_a_layer, lora_b_layer)
            self._name_map[module_name] = safe_key
            # print(f"  - Built LoRA module for: {module_name} -> safe_key: {safe_key} (adapter: {'<none>' if '.' not in a_key else adapter}, in: {in_features}, r: {rank}, out: {out_features})")
 
        self._modules.eval() # 设置为评估模式
        print("RemoteLoraGuest initialized successfully.")

    @torch.no_grad() # 推理时不需要计算梯度
    def forward(self, module_name: str, x: torch.Tensor):
        """
        执行指定LoRA模块的前向传播。
        
        Args:
            module_name (str): Host端告知的、需要计算的原始模块名。
            x (torch.Tensor): Host端发送过来的输入张量。
        
        Returns:
            torch.Tensor: 计算出的增量 delta_h。
        """
        safe_key = self._name_map.get(module_name)
        if safe_key is None:
            # 兼容：如果 host 传入的就是安全键（罕见场景），直接使用
            if module_name in self._modules:
                safe_key = module_name
            else:
                raise ValueError(f"Unknown LoRA module name: {module_name}")

        # 确保输入张量的数据类型与模型权重一致
        lora_layers = self._modules[safe_key]
        dtype = lora_layers[0].weight.dtype
        x = x.to(dtype)

        # 计算 delta_h = B(A(x)) * scaling
        delta_h = lora_layers(x) * self._scaling
        
        return delta_h


class SliceLinear(LoraLinear):
    """
    一个特殊的LoRA线性层，它只在Host端执行基础模型部分的计算。
    LoRA增量的计算则通过ivshmem委托给Guest端。
    """
    def __init__(self, target: LoraLinear, module_name: str, shm):
        active_adapter = "default"
        
        r_value = target.r[active_adapter]
        alpha_value = target.lora_alpha[active_adapter]
        dropout_module = target.lora_dropout[active_adapter]
        dropout_value = dropout_module.p

        super().__init__(
            base_layer=target.base_layer,
            adapter_name=active_adapter,
            r=r_value,
            lora_alpha=alpha_value,
            lora_dropout=dropout_value,
            fan_in_fan_out=target.fan_in_fan_out,
        )

        self.base_layer = target.base_layer
        self.module_name = module_name
        self.shm = shm

    def forward(self, x: torch.Tensor):
        # host_start = time.time()
        active_adapters = self.active_adapter if isinstance(self.active_adapter, (list, tuple)) else [self.active_adapter]

        # 如果禁用 adapter，或所有激活 adapter 都不在本层 lora_A 中，就跳过 LoRA 部分
        if self.disable_adapters or all(a not in self.lora_A for a in active_adapters):
            print("非LoRA层，直接本地计算")
            # base_end = time.time()
            return self.base_layer(x)

        result = self.base_layer(x)
        # base_end = time.time()

        tensor_bytes = ic.tensor2bytes(x)
        request_blocks = ic.tensor_bytes_and_module_name2blocks(tensor_bytes, msg_id=0, module_name=self.module_name)
        
        # host_write_start = time.time()
        ic.write_blocks(self.shm, request_blocks, "host")
        # host_write_end = time.time()

        response_blocks = []
        host_read_start, host_read_end = 0, 0
        while True:
            # host_read_start = time.time()
            response_blocks = ic.read_blocks(self.shm, "host")
            if len(response_blocks) > 0:
                # host_read_end = time.time()
                break
            time.sleep(RR)
        delta_h_bytes, _ = ic.blocks2tensor_bytes_and_module_name(response_blocks)
        delta_h = ic.bytes2tensor(delta_h_bytes, use_gpu=torch.cuda.is_available())
        
        delta_h = delta_h.to(result.device, dtype=result.dtype)
        result += delta_h
        # host_end = time.time()
        # with open("log_host.txt", "a") as log_f:
        #     # host传输时间 host的forward函数中，除了传输和等待guest的时间 host等待guest的时间
        #     log_f.write(f"{(host_write_end - host_write_start + host_read_end - host_read_start):.6f} {(host_end - host_read_end + host_write_start - host_start):6f} {(host_read_start - host_write_end):6f} {host_write_end} {host_read_start}\n")
        
        return result

def replace_lora_layers(model: nn.Module, shm):
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)

            new_module = SliceLinear(module, name, shm)
            setattr(parent_module, child_name, new_module)
            replaced_count += 1
            # print(f"  - Replaced '{name}' with SliceLinear.")
    
    print(f"Replacement complete. Total layers replaced: {replaced_count}")
    return model


shared_state = {
    'request_id': 0,
    'module_name': None,
    'input_tensor': None,
    'exit_signal': False # 退出信号
}

def guest_worker(worker_id, guest_lora_model, condition, output_queue):
    global shared_state
    guest_forward_time = 0.0    
    # 追踪该 worker 已经处理的请求 ID，避免重复处理
    last_processed_id = 0
    # if worker_id == 0:
    #     log_f = open("log_guest_worker_0.txt", "a")
    
    while True:
        # 局部变量，用于存储本次处理的数据
        module_name, input_tensor = None, None
        
        with condition:
            # 循环等待：只有当 request_id 大于该 worker 上次处理的 ID 时才退出等待
            # 或者收到退出信号时退出等待
            while shared_state['request_id'] == last_processed_id and not shared_state['exit_signal']:
                condition.wait()
            
            # 检查退出信号
            if shared_state['exit_signal']:
                print(f"Worker {worker_id} - forward时间累计 {guest_forward_time:.6f} 秒")
                break
            
            # 确定是新数据：更新已处理 ID，并读取共享数据引用
            last_processed_id = shared_state['request_id']
            module_name = shared_state['module_name']
            input_tensor = shared_state['input_tensor']
        
        guest_forward_start = time.time()
        # 将开始执行的时间戳写入log_guest_worker_0.txt
        # if worker_id == 0:
        #     log_f.write(f"{guest_forward_start}\n")
        delta_h = guest_lora_model.forward(module_name, input_tensor)
        guest_forward_end = time.time()
        # if worker_id == 0:
        #     log_f.write(f"{guest_forward_end}\n")
        output_queue.put((worker_id, delta_h))

        guest_forward_time += guest_forward_end - guest_forward_start


def guest_main(set_multi_thread=False):
    open("log_guest.txt", "w").close()
    open("log_tensor2bytes.txt", "w").close()
    open("log_tensor_bytes_and_module_name2blocks.txt", "w").close()
    open("log_blocks2tensor_bytes_and_module_name.txt", "w").close()
    open("log_bytes2tensor.txt", "w").close()
    open("log_guest_master.txt", "w").close()
    open("log_guest_worker_0.txt", "w").close()
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
        ic.write_ret_uint8(shm, 1) # 告诉host，guest端准备好了
        print("[GUEST] LoRA layers initialized. Waiting for requests...")

        # 单线程模式
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
        else:    
            # 多线程模式
            thread_num = 32

            global shared_state
            # Condition 对象用于同步：主线程通知，工作线程等待
            condition = threading.Condition()
            output_queue = queue.Queue()

            # 启动工作线程
            workers = []
            for i in range(thread_num):
                t = threading.Thread(target=guest_worker, 
                                    args=(i, guest_lora_model, condition, output_queue))
                t.daemon = True
                t.start()
                workers.append(t)
                
            total_broadcast_overhead = 0.0 # 记录主线程广播（写共享内存和通知）的开销
            total_request_latency = 0.0    # 记录从请求开始到收到第一个结果的总时间

            # 将notify workers的时间戳写入log_guest_master.txt
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
                        log_f.write(f"{request_end_time}\n")

                        total_request_latency += (request_end_time - request_start_time)
                        
                        worker_id, delta_h = results[0]
                        
                        response_bytes = ic.tensor2bytes(delta_h)
                        response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
                        ic.write_blocks(shm, response_blocks, "guest")
                    else:
                        time.sleep(RR)
                    
            except KeyboardInterrupt:
                print(f"主线程广播（写共享内存和通知）总开销: {total_broadcast_overhead:.6f} 秒")
                print(f"请求处理总延迟（从开始广播到收到所有结果）: {total_request_latency:.6f} 秒")
                
                with condition:
                    shared_state['exit_signal'] = True
                    condition.notify_all()
                    
                for t in workers:
                    t.join()
                
                log_f.close()

def check_lora_weights_zero(adapter_path):
    """检查适配器权重是否全为零"""
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            return False, []
    
    # 加载适配器权重
    try:
        if adapter_weights_path.endswith('.safetensors'):
            state_dict = load_file(adapter_weights_path)
        else:
            state_dict = torch.load(adapter_weights_path, map_location='cpu')
    except:
        return False, []
    
    unzero_modules = []
    
    # 检查每个LoRA权重是否全为零
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
    """特判llama的模块命名问题"""
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            return False, []
    
    # 加载适配器权重
    try:
        if adapter_weights_path.endswith('.safetensors'):
            state_dict = load_file(adapter_weights_path)
        else:
            state_dict = torch.load(adapter_weights_path, map_location='cpu')
    except:
        return False, []
    
    lora_modules = []
    
    # 检查每个LoRA权重是否全为零
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
    # base_model = AutoModelForQuestionAnswering.from_pretrained(
        base_model_dir,
        dtype=DEFAULT_DTYPE,
        device_map=None,
    )
    base_model.to(device)
    # print(f"base model架构：{base_model}")
    
    # 剪枝lora
    if PRUNED:
        _, unzero_modules = check_lora_weights_zero(lora_model_dir)
        # print(f"非零LoRA模块有: {unzero_modules}")
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
                    #print(f"加载LoRA权重: {name}")
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]

        # model.load_state_dict(state_dict, strict=False)
        # print(f"完整模型架构: {model}")
    # 全量lora
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
        # print(f"LoRA模块有: {lora_modules}")
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
                    #print(f"加载LoRA权重: {name}")
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
        # 1) 先在模型还保留 LoraLinear 时序列化 LoRA 权重和配置并写入共享内存
        lora_weights_bytes, lora_config_bytes = ic.lora_weight_config2bytes(model)
        lora_weights_block_num = (len(lora_weights_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_weight_blocks = ic.bytes2blocks(lora_weights_bytes, msg_id=lora_weights_block_num, force_is_not_last=True)
        lora_config_blocks = ic.bytes2blocks(lora_config_bytes, msg_id=lora_weights_block_num)
        print(f"msg_id是{lora_weights_block_num}")
        print(f"两段blocks的长度分别是{len(lora_weight_blocks)}和{len(lora_config_blocks)}")
        packed_blocks = lora_weight_blocks + lora_config_blocks
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        ic.write_blocks(shm, packed_blocks, "host")

        # 2) 在把权重发送完毕并通知 guest 后，再把本地的 LoraLinear 替换为委托层
        model = replace_lora_layers(model, shm)
    
    # 读到guest端准备好的信号后，开始推理
    while True:
        if ic.read_ret_uint8(shm) == 1:
            break
        else:
            time.sleep(0.01)
            
    init_end = time.time()
    init_time = init_end - init_start
    print("[HOST] 推理开始")
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
    # print(f"[HOST] 初始化时间: {init_time:.6f} 秒")
    print(f"[HOST] 生成时间: {gen_end - gen_start:.6f} 秒")
    print(f"[HOST] 输出token数: {len(output_ids[0])}")
    # print(f"[HOST] 总时间: {total_time:.6f} 秒")

def test_host():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先在默认位置加载 base_model（不使用 device_map="auto"），然后显式移动到目标 device
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=DEFAULT_DTYPE, # host上运行会显示torch_dtype已废弃，应该用dtype；但是guest上不用torch_dtype就会报错
        device_map=None,
    )
    # 剪枝lora
    if PRUNED:
        _, unzero_modules = check_lora_weights_zero(lora_model_dir)
        # print(f"非零LoRA模块有: {unzero_modules}")
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
                    #print(f"加载LoRA权重: {name}")
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]

        # model.load_state_dict(state_dict, strict=False)
        # print(f"完整模型架构: {model}")
    # 全量lora
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
        # print(f"LoRA模块有: {lora_modules}")
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
                    #print(f"加载LoRA权重: {name}")
                    for state_dict_key in state_dict.keys():
                        if new_name2 in state_dict_key:
                            param.data = state_dict[state_dict_key]
        # model.load_state_dict(state_dict, strict=False)

    model.to(device=device, dtype=DEFAULT_DTYPE)
    model.eval()

    # 诊断：PeftModel 与 LoRA 层计数，以及参数设备分布
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
    print(f"[HOST-ONLY] 推理时间: {gen_end - gen_start:.6f} 秒")
    print(f"[HOST-ONLY] 输出token数: {len(output_ids[0])}")


# host write, guest read, guest write, host read。这怎么会导致数据不一致？
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