import io
import torch
import numpy as np
import json
import struct
import time
import ctypes


# header，block（header+payload）的定义，序列化和反序列化的函数
LOCK_OFFSET = 0  # 读写锁占用1字节
READ_RET_OFFSET = 1
HOST_GUEST_OFFSET = 2
'''
由于分层推理时，生成多token需要每次回到模型开头，再重复推理流程。所以最后一层输出之后需要将结果返回到推理开始的那一端（通过ivshmem），
而且达到输出token上限（或者eos之类的）而得到最终输出之后，也需要交给推理开始的那一端解码输出。所以在内存中应该再拿一部分出来，作为“推
理开始的那一端此时应该读取内存中的内容”的标识。所以我添加常量READ_RET_OFFSET=1，即第二个字节的共享内存拿来做这个标识。
HOST_GUEST这个标识不是简单的0或1，而是记录了推理的start_pos（为了生成多token），uint8，一旦开始推理的那一端检测到这个标识被加了1（由推理的末端
写入），就判断这个值是否达到max token数，达到了就解码输出，否则推下一轮。写是由结束端读取然后加一，写端也要判断是否达到max token数，
以区分返回的内容是token ids还是下一轮推理所用到的隐藏状态（尽管都是tensor）。

当推理流程是guest-host-guest-host时，第二轮guest（开始端）还没发第一批的输出呢，host就自己读到脏东西了
这是因为host发完output后马上进了下一轮，又读取了SHM，自己读自己
要添加一个设计，拿第三个字节来作为里面的内容是host还是guest写入的，0为host，1为guest
'''
BLOCK_SIZE = 4096 + 9
HEADER_SIZE = 9
PAYLOAD_SIZE = 4096
MAX_BLOCK_NUM = 4087 # (16*1024*1024 - 1)//BLOCK_SIZE，1是读写锁使用的1个字节

# <即小端法，I = uint32, H = uint16, B = uint8
# msg_id uint32 一层输出的所有tensor共用一个msg_id
# seq_id uint16 该序列的第几个block
# is_last uint8 该block是否是最后一个
# payload_len uint16 该block的payload长度，上限是4096，所以uint16就够了
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

def clear_shm(shm): # 清空共享内存
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(shm, HOST_GUEST_OFFSET + 1)), 0, 16)


# 一层的输出->字节
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

# blocks->tensor的字节序列和module_name
def blocks2tensor_bytes_and_module_name(blocks: list) -> tuple:
    # func_start = time.time()
    # 按seq_id排序
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

# tensor的字节序列->给下一层的tensor
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





# 共享内存通信逻辑：不冲突地从共享内存读写，利用共享内存的第一个字节管理
# 虽然host和guest访问shared memory的方式不同，但已经在run_host/guest中处理，将其作为字节数组shm传入
# 一次锁住整块共享内存，因为实际场景下推完一层交给下一层是单向的，也只有组装起完整的tensor才能开始推下一层。

# 虽然不是原子的，但是同时只有一方写，另一方只读，所以只要写了一点，就不可能在写完之前读到，再不就是读空。反之亦然
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
        copy_time = 0.0 # 记录复制内存的时间
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
    should_clear = True # 空的或没有权限读，就不能清空。只有读到了才能清空
    have_blocks = False
    try:
        blocks = []
        offset = HOST_GUEST_OFFSET + 1
        copy_time = 0.0 # 记录复制内存的时间
        while offset + HEADER_SIZE <= len(shm): # 实际上就是offset <= len(shm)，这么写保险些而已（下一行）
            if role == "host" and read_host_guest_uint8(shm) == 0: # host写入的不能host读
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
                # 可能读完了整个共享内存，仍没有读完整个tensor，这时要清空共享内存并从头开始，等1秒让写方接着写
                if len(blocks) > 0 and len(blocks) % MAX_BLOCK_NUM == 0:
                    clear_shm(shm)
                    # release_lock(shm)
                    time.sleep(2)
                    # acquire_lock(shm)
                    offset = HOST_GUEST_OFFSET + 1
        if have_blocks:
            # print(f"{role} 读，读取block数量{len(blocks)}")
            pass
        return blocks
    finally:
        if should_clear:
            clear_shm(shm)
        # release_lock(shm)



# 初始化时guest用来反序列化得到lora相关权重或配置字节序列的函数，和针对tensor的处理区分开
# 从完整模型中提取lora权重，获取lora配置，序列化权重和配置
def lora_weight_config2bytes(model):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.cpu().clone()
    lora_config = model.peft_config['default'].to_dict()

    # 遍历配置字典，将所有set类型转换为list类型，否则json序列化会报错
    for key, value in lora_config.items():
        if isinstance(value, set):
            lora_config[key] = list(value)

    buffer = io.BytesIO()
    torch.save(lora_state_dict, buffer)
    lora_weights_bytes = buffer.getvalue()

    lora_config_bytes = json.dumps(lora_config).encode('utf-8')
    # 打印两个字节序列的大小（单位为KB）
    print(f"LoRA weights size: {len(lora_weights_bytes) / 1024:.2f} KB")
    print(f"LoRA config size: {len(lora_config_bytes) / 1024:.2f} KB")
    return lora_weights_bytes, lora_config_bytes

# 由于将lora权重和配置一次传输，所以第一批（权重）的块不能设置is_last=1，需要特判
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
    # 按seq_id排序
    blocks.sort(key=lambda b: struct.unpack('<H', b[4:6])[0])
    payloads = [b[HEADER_SIZE:HEADER_SIZE + struct.unpack('<H', b[7:9])[0]] for b in blocks]
    payload_bytes = b''.join(payloads)
    return payload_bytes

def bytes2lora_weight_config(lora_weights_bytes, lora_config_bytes):
    # 反序列化权重
    buffer = io.BytesIO(lora_weights_bytes)
    lora_state_dict = torch.load(buffer, map_location='cpu')

    # 反序列化配置
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