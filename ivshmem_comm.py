import io
import torch
import numpy as np
import json
import struct
import time
import ctypes

LOCK_OFFSET = 0
READ_RET_OFFSET = 1
HOST_GUEST_OFFSET = 2

BLOCK_SIZE = 4096 + 9
HEADER_SIZE = 9
PAYLOAD_SIZE = 4096
MAX_BLOCK_NUM = 4087 # (16*1024*1024 - 1)//BLOCK_SIZE

# < = small endian，I = uint32, H = uint16, B = uint8
# msg_id uint32: output tensor from the same forward pass has the same msg_id
# seq_id uint16: the sequence number of this block in the whole tensor bytes
# is_last uint8
# payload_len uint16: atmost 4096 bytes
BLOCK_HEADER_FORMAT = '<IHBH'

# 辅助函数
def get_msg_id(block: bytes) -> int:
    # 解析block的header，获取msg_id
    header = block[:HEADER_SIZE]
    msg_id, _, _, _ = struct.unpack(BLOCK_HEADER_FORMAT, header)
    return msg_id

def read_ret_uint8(shm):
    return int.from_bytes(shm[READ_RET_OFFSET:READ_RET_OFFSET + 1], 'little')

def write_ret_uint8(shm, value):
    if value < 0 or value > 255:
        value = value & 0xFF
    shm[READ_RET_OFFSET:READ_RET_OFFSET + 1] = value.to_bytes(1, 'little')

def read_host_guest_uint8(shm):
    return int.from_bytes(shm[HOST_GUEST_OFFSET:HOST_GUEST_OFFSET + 1], 'little')

def write_host_guest_uint8(shm, value):
    if value < 0 or value > 255:
        value = value & 0xFF
    shm[HOST_GUEST_OFFSET:HOST_GUEST_OFFSET + 1] = value.to_bytes(1, 'little')

def clear_shm(shm):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(shm, HOST_GUEST_OFFSET + 1)), 0, 16)


def tensor2bytes(tensor: torch.Tensor) -> bytes:
    # func_start = time.time()
    t = tensor.detach()
    orig_dtype = str(t.dtype)  # record original torch dtype as string

    # Move to CPU for numpy conversion. If dtype is bfloat16 (unsupported by numpy),
    # convert to float32 on CPU first.
    t_cpu = t.cpu()
    if t_cpu.dtype == torch.bfloat16:
        # numpy doesn't support bfloat16 -> convert to float32 for storage
        t_cpu = t_cpu.to(torch.float32)
        meta_dtype = 'float32'
    else:
        # Use numpy-compatible dtype string
        meta_dtype = str(t_cpu.numpy().dtype)

    np_array = t_cpu.numpy()
    meta = {
        'shape': list(np_array.shape),
        'dtype': meta_dtype,        # dtype used for the raw bytes (numpy dtype string)
        'orig_torch_dtype': orig_dtype,  # original torch dtype (e.g. 'torch.bfloat16')
    }
    meta_bytes = json.dumps(meta).encode('utf-8')
    meta_len = len(meta_bytes).to_bytes(4, 'little')  # prepend 4-byte length
    # func_end = time.time()
    # with open("log_tensor2bytes.txt", "a") as log_f:
    #     log_f.write(f"{(func_end - func_start):.6f}\n")
    return meta_len + meta_bytes + np_array.tobytes()



def tensor_bytes_and_module_name2blocks(tensor_bytes: bytes, msg_id: int, module_name: str = '') -> list:
    # func_start = time.time()
    name_bytes = module_name.encode('utf-8')
    name_len_bytes = struct.pack('<I', len(name_bytes)) # '<I' = 4-byte unsigned int
    total_data_bytes = name_len_bytes + name_bytes + tensor_bytes
    blocks = []
    total_len = len(total_data_bytes)
    num_blocks = (total_len + PAYLOAD_SIZE - 1) // PAYLOAD_SIZE

    for seq_id in range(num_blocks):
        start = seq_id * PAYLOAD_SIZE
        end = min(start + PAYLOAD_SIZE, total_len)
        payload = total_data_bytes[start:end]
        payload_len = len(payload)
        is_last = 1 if seq_id == num_blocks - 1 else 0

        header = struct.pack(BLOCK_HEADER_FORMAT, msg_id, seq_id, is_last, payload_len)
        assert len(header) == HEADER_SIZE, 'Header must be exactly 9 bytes'
        blocks.append(header + payload)
    # func_end = time.time()
    # with open("log_tensor_bytes_and_module_name2blocks.txt", "a") as log_f:
    #     log_f.write(f"{(func_end - func_start):.6f}\n")
    return blocks

def blocks2tensor_bytes_and_module_name(blocks: list) -> tuple:
    # func_start = time.time()
    blocks.sort(key=lambda b: struct.unpack('<H', b[4:6])[0])
    payloads = [b[HEADER_SIZE:HEADER_SIZE + struct.unpack('<H', b[7:9])[0]] for b in blocks]
    total_data_bytes = b''.join(payloads)
    name_len = struct.unpack('<I', total_data_bytes[:4])[0]
    name_end = 4 + name_len
    module_name_bytes = total_data_bytes[4:name_end]
    module_name = module_name_bytes.decode('utf-8')
    tensor_bytes = total_data_bytes[name_end:]
    # func_end = time.time()
    # with open("log_blocks2tensor_bytes_and_module_name.txt", "a") as log_f:
    #     log_f.write(f"{(func_end - func_start):.6f}\n")
    return tensor_bytes, module_name

def bytes2tensor(data: bytes, use_gpu: bool = False) -> torch.Tensor:
    # func_start = time.time()
    meta_len = int.from_bytes(data[:4], 'little')
    meta = json.loads(data[4:4+meta_len].decode('utf-8'))
    tensor_data = data[4+meta_len:]

    # Build numpy array from raw bytes
    np_dtype = meta['dtype']
    np_array = np.frombuffer(tensor_data, dtype=np_dtype).reshape(meta['shape'])
    # Create a CPU tensor first
    t = torch.from_numpy(np_array.copy())

    orig_torch_dtype = meta.get('orig_torch_dtype', None)
    # func_end = time.time()
    # with open("log_bytes2tensor.txt", "a") as log_f:
    #     log_f.write(f"{(func_end - func_start):.6f}\n")
    # If original was bfloat16 and user wants GPU and CUDA available, cast back on GPU
    if use_gpu and orig_torch_dtype == 'torch.bfloat16' and torch.cuda.is_available():
        return t.cuda().to(torch.bfloat16)
    if use_gpu:
        return t.cuda()
    return t

def acquire_lock(shm):
    while True:
        if shm[LOCK_OFFSET] == 0:
            shm[LOCK_OFFSET] = 1
            break
        time.sleep(0.00001)  # Sleep 0.1ms to reduce busy-wait

def release_lock(shm):
    shm[LOCK_OFFSET] = 0


def write_blocks(shm, blocks, role):
    clear_shm(shm)
    # acquire_lock(shm)
    try:
        # write_host_guest_uint8(shm, 0 if role == "host" else 1)
        
        offset = HOST_GUEST_OFFSET + 1
        block_count = 0
        copy_time = 0.0
        for block in blocks:
            if block_count > 0 and block_count % MAX_BLOCK_NUM == 0:
                # release_lock(shm)
                write_host_guest_uint8(shm, 0 if role == "host" else 1)
                time.sleep(1)
                # acquire_lock(shm)
                offset = HOST_GUEST_OFFSET + 1
            shm[offset:offset+len(block)] = block
            offset += BLOCK_SIZE
            block_count += 1
    finally:
        # release_lock(shm)
        write_host_guest_uint8(shm, 0 if role == "host" else 1)

def read_blocks(shm, role):
    # acquire_lock(shm)
    should_clear = True # can't clear if nothing read or no permission to read
    have_blocks = False
    try:
        blocks = []
        offset = HOST_GUEST_OFFSET + 1
        copy_time = 0.0
        while offset + HEADER_SIZE <= len(shm):
            if role == "host" and read_host_guest_uint8(shm) == 0:
                should_clear = False
                continue
            if role == "guest" and read_host_guest_uint8(shm) == 1:
                should_clear = False
                continue
            header = shm[offset:offset+HEADER_SIZE]
            if all(b == 0 for b in header):
                should_clear = False
                continue
            have_blocks = True
            msg_id, seq_id, is_last, payload_len = struct.unpack(BLOCK_HEADER_FORMAT, header)
            payload_start = offset + HEADER_SIZE
            payload_end = payload_start + payload_len
            payload = shm[payload_start:payload_end]
            full_block = header + payload
            blocks.append(full_block)
            offset += BLOCK_SIZE
            if is_last:
                break
            else:
                # havn't read the full tensor even reach the end of shm
                # so clear shm and start from beginning, wait 1 second for writer to write more
                if len(blocks) > 0 and len(blocks) % MAX_BLOCK_NUM == 0:
                    clear_shm(shm)
                    # release_lock(shm)
                    time.sleep(2)
                    # acquire_lock(shm)
                    offset = HOST_GUEST_OFFSET + 1
        if have_blocks:
            pass
        return blocks
    finally:
        if should_clear:
            clear_shm(shm)
        # release_lock(shm)


def lora_weight_config2bytes(model):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.cpu().clone()
    lora_config = model.peft_config['default'].to_dict()

    for key, value in lora_config.items():
        if isinstance(value, set):
            lora_config[key] = list(value)

    buffer = io.BytesIO()
    torch.save(lora_state_dict, buffer)
    lora_weights_bytes = buffer.getvalue()

    lora_config_bytes = json.dumps(lora_config).encode('utf-8')
    print(f"LoRA weights size: {len(lora_weights_bytes) / 1024:.2f} KB")
    print(f"LoRA config size: {len(lora_config_bytes) / 1024:.2f} KB")
    return lora_weights_bytes, lora_config_bytes


def bytes2blocks(data_bytes: bytes, msg_id: int, force_is_not_last: bool=False) -> list:
    blocks = []
    total_len = len(data_bytes)
    num_blocks = (total_len + PAYLOAD_SIZE - 1) // PAYLOAD_SIZE

    for seq_id in range(num_blocks):
        start = seq_id * PAYLOAD_SIZE
        end = min(start + PAYLOAD_SIZE, total_len)
        payload = data_bytes[start:end]
        payload_len = len(payload)
        is_last = 1 if seq_id == num_blocks - 1 and force_is_not_last is False else 0

        header = struct.pack(BLOCK_HEADER_FORMAT, msg_id, seq_id, is_last, payload_len)
        assert len(header) == HEADER_SIZE, 'Header must be exactly 9 bytes'
        blocks.append(header + payload)

    return blocks

def blocks2bytes(blocks: list) -> bytes:
    blocks.sort(key=lambda b: struct.unpack('<H', b[4:6])[0])
    payloads = [b[HEADER_SIZE:HEADER_SIZE + struct.unpack('<H', b[7:9])[0]] for b in blocks]
    payload_bytes = b''.join(payloads)
    return payload_bytes

def bytes2lora_weight_config(lora_weights_bytes, lora_config_bytes):
    buffer = io.BytesIO(lora_weights_bytes)
    lora_state_dict = torch.load(buffer, map_location='cpu')

    lora_config = json.loads(lora_config_bytes.decode('utf-8'))
    return lora_state_dict, lora_config



if __name__ == '__main__':
    # print("===== CPU Test =====")
    # cpu_tensor = torch.randn(100, 100, 100)
    # cpu_bytes = serialize_tensor(cpu_tensor)
    # cpu_blocks = split_tensor_bytes(cpu_bytes, 2)
    # cpu_assembled = assemble_blocks(cpu_blocks)
    # cpu_reconstructed = deserialize_tensor(cpu_assembled, use_gpu=False)
    # print(f"CPU tensor shape after round trip: {cpu_reconstructed.shape}")

    if torch.cuda.is_available():
        print("===== GPU Test =====")
        gpu_tensor = torch.randn(100, 100, 100).cuda()
        gpu_bytes = tensor2bytes(gpu_tensor)
        gpu_blocks = tensor_bytes_and_module_name2blocks(gpu_bytes, 3)
        gpu_assembled = blocks2tensor_bytes_and_module_name(gpu_blocks)
        gpu_reconstructed = bytes2tensor(gpu_assembled, use_gpu=True)
        print(f"GPU tensor shape after round trip: {gpu_reconstructed.shape}")