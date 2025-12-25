__package__ = "pkus".model_def

import time, torch
from pathlib import Path
from torch import nn
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
import ivshmem_comm as ic

RR = 0.0001 # round-robin sleep interval

class GuestLoraModel:
    def __init__(self, lora_state_dict, lora_config_dict):
        self.lora_config = LoraConfig(**lora_config_dict)
        self._modules = nn.ModuleDict()
        self._name_map = {}
        self._scaling = self.lora_config.lora_alpha / self.lora_config.r

        lora_a_keys = [k for k in lora_state_dict if (".lora_A." in k and k.endswith(".weight")) or k.endswith("lora_A.weight")]

        for a_key in lora_a_keys:
            if ".lora_A." in a_key and a_key.endswith(".weight"):
                module_name, tail = a_key.split(".lora_A.", 1)
                adapter = tail[:-len(".weight")]  # e.g. "default"
                b_key = f"{module_name}.lora_B.{adapter}.weight"
            else:
                module_name = a_key.removesuffix(".lora_A.weight")
                b_key = f"{module_name}.lora_B.weight"

            if b_key not in lora_state_dict:
                print(f"  - Warning: Corresponding LoRA B key not found for '{a_key}', expected '{b_key}'. Skipping.")
                continue

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

            lora_a_layer = nn.Linear(in_features, rank, bias=False)
            lora_b_layer = nn.Linear(rank, out_features, bias=False)

            with torch.no_grad():
                lora_a_layer.weight.data.copy_(lora_a_weight)
                lora_b_layer.weight.data.copy_(lora_b_weight)

            safe_key = module_name.replace('.', '-')
            self._modules[safe_key] = nn.Sequential(lora_a_layer, lora_b_layer)
            self._name_map[module_name] = safe_key
            # print(f"  - Built LoRA module for: {module_name} -> safe_key: {safe_key} (adapter: {'<none>' if '.' not in a_key else adapter}, in: {in_features}, r: {rank}, out: {out_features})")
 
        self._modules.eval()
        print("RemoteLoraGuest initialized successfully.")

    @torch.no_grad() # 推理时不需要计算梯度
    def forward(self, module_name: str, x: torch.Tensor):
        safe_key = self._name_map.get(module_name)
        if safe_key is None:
            if module_name in self._modules:
                safe_key = module_name
            else:
                raise ValueError(f"Unknown LoRA module name: {module_name}")

        lora_layers = self._modules[safe_key]
        dtype = lora_layers[0].weight.dtype
        x = x.to(dtype)

        delta_h = lora_layers(x) * self._scaling
        return delta_h


class SliceLinear(LoraLinear):
    """
    Do basic linear computation on Host side, and delegate LoRA incremental computation to Guest side via ivshmem.
    args:
        target: Original LoraLinear layer from PEFT model.
        module_name: Name of the module in the model, used for identification in communication.
        shm: Shared memory object for ivshmem communication.
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

        # skip LoRA computation if adapters are disabled or none are active for this layer
        if self.disable_adapters or all(a not in self.lora_A for a in active_adapters):
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