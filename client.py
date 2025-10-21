import sys, os, json, time, mmap, torch, argparse
from pathlib import Path
from torch import nn
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.layer import Linear as LoraLinear
import ivshmem_comm as ic

BASE_MODEL_DIR = "./model/llama-3-1b-instruct"
LORA_MODEL_DIR = "./model/llama-3-1b-lora"
HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

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
        # print("1. 在Host本地计算基础模型的输出")
        host_start = time.time()
        active_adapters = self.active_adapter if isinstance(self.active_adapter, (list, tuple)) else [self.active_adapter]

        # 如果禁用 adapter，或所有激活 adapter 都不在本层 lora_A 中，就跳过 LoRA 部分
        if self.disable_adapters or all(a not in self.lora_A for a in active_adapters):
            print("非LoRA层，直接本地计算")
            return self.base_layer(x)

        result = self.base_layer(x)
        host_end = time.time()

        # print("2. 将LoRA部分的计算外包给Guest")
        tensor_bytes = ic.tensor2bytes(x)
        msg_id = int(time.time()) 
        request_blocks = ic.tensor_bytes_and_module_name2blocks(tensor_bytes, msg_id=0, module_name=self.module_name)
        
        # print("3. 发送请求到共享内存")
        # print(f"请求包含 {len(request_blocks)} 个块")
        host_write_start = time.time()
        ic.write_blocks(self.shm, request_blocks, "host")
        host_write_end = time.time()

        # print("4. 等待并接收Guest的响应")
        response_blocks = []
        host_read_start, host_read_end = 0, 0
        while True:
            host_read_start = time.time()
            response_blocks = ic.read_blocks(self.shm, "host")
            if len(response_blocks) > 0:
                host_read_end = time.time()
                break
            time.sleep(0.001)
        # print(f"响应包含 {len(response_blocks)} 个块")
        # print("5. 解析响应")
        delta_h_bytes, _ = ic.blocks2tensor_bytes_and_module_name(response_blocks)
        delta_h = ic.bytes2tensor(delta_h_bytes, use_gpu=torch.cuda.is_available())
        
        # print("6. 合并结果")
        delta_h = delta_h.to(result.device, dtype=result.dtype)
        result += delta_h
        guest_end = time.time()
        with open("log_host.txt", "a") as log_f:
            log_f.write(f"{guest_end - host_end:.6f} {host_write_end - host_write_start:.6f} {host_read_end - host_read_start:.6f}\n")
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
            print(f"  - Replaced '{name}' with SliceLinear.")
    
    print(f"Replacement complete. Total layers replaced: {replaced_count}")
    return model


def guest_main():
    open("log_guest.txt", "w").close()
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        while True:
            blocks = ic.read_blocks(shm, "guest")
            if len(blocks) > 0:
                split_point = ic.get_msg_id(blocks[0])
                lora_weight_blocks = blocks[:split_point]
                lora_config_blocks = blocks[split_point:]
                break
            else:
                time.sleep(0.001)

        serialized_weights = ic.blocks2bytes(lora_weight_blocks)
        serialized_config = ic.blocks2bytes(lora_config_blocks)
        print(f"LoRA weights size: {len(serialized_weights) / 1024:.2f} KB")
        print(f"LoRA config size: {len(serialized_config) / 1024:.2f} KB")
        lora_state_dict, lora_config_dict = ic.bytes2lora_weight_config(serialized_weights, serialized_config)
        guest_lora_model = GuestLoraModel(lora_state_dict, lora_config_dict)
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 1) # 告诉host，guest端准备好了
        print("[GUEST] LoRA layers initialized. Waiting for requests...")
        
        while True:
            guest_read_start = time.time()
            request_blocks = ic.read_blocks(shm, "guest")
            if len(request_blocks) > 0:
                guest_read_end = time.time()
                tensor_bytes, module_name = ic.blocks2tensor_bytes_and_module_name(request_blocks)
                input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)
                # print(f"[GUEST] Received request for module: {module_name}")

                # 执行LoRA前向传播计算
                guest_forward_start = time.time()
                delta_h = guest_lora_model.forward(module_name, input_tensor)
                guest_forward_end = time.time()
                # 准备并发送响应
                response_bytes = ic.tensor2bytes(delta_h)
                msg_id = int(time.time())
                response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
                guest_write_start = time.time()
                ic.write_blocks(shm, response_blocks, "guest")
                guest_write_end = time.time()
                # print(f"[GUEST] Sent response for module: {module_name}")
                with open("log_guest.txt", "a") as log_f:
                    log_f.write(f"{guest_forward_end - guest_forward_start:.6f} {guest_write_end - guest_write_start:.6f} {guest_read_end - guest_read_start:.6f}\n")
            else:
                time.sleep(0.001)

def host_main():
    open("log_host.txt", "w").close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_DIR,
    )
    model.eval()
    
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        model = replace_lora_layers(model, shm)
        lora_weights_bytes, lora_config_bytes = ic.lora_weight_config2bytes(model)
        lora_weights_block_num = (len(lora_weights_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_weight_blocks = ic.bytes2blocks(lora_weights_bytes, msg_id=lora_weights_block_num, force_is_not_last=True)
        lora_config_blocks = ic.bytes2blocks(lora_config_bytes, msg_id=lora_weights_block_num)
        print(f"msg_id是{lora_weights_block_num}")
        print(f"两段blocks的长度分别是{len(lora_weight_blocks)}和{len(lora_config_blocks)}")
        # 拼接两组块，发送lora权重和配置
        packed_blocks = lora_weight_blocks + lora_config_blocks
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        ic.write_blocks(shm, packed_blocks, "host")
        
    
    # 读到guest端准备好的信号后，开始推理
    while True:
        if ic.read_ret_uint8(shm) == 1:
            break
        else:
            time.sleep(0.01)
            
    print("[HOST] 推理开始")
    # prompt = "How many states does the US have?"
    prompt = "Say \"Hello\", do not include other words."
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,             # 启用采样以获得更自然的结果
            temperature=0.7,            # 采样温度
            top_p=0.9,                  # Top-p 采样
            repetition_penalty=1.1,     # 惩罚重复
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Model Inference")
    parser.add_argument("--role", choices=["host", "guest"], help="Role of the client (host or guest)")
    args = parser.parse_args()
    client_role = args.role
    if client_role == "host":
        host_main()
    else:
        guest_main()