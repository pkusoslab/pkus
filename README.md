# 🔧 Prerequisites
- **Python**: 3.12 or higher
- **Git**: For repository cloning and submodule management
- **Cuda** compilation tools: 12.0 or higher
- **Pytorch**: 2.7.0+cu128 or higher
- **QEMU** emulator: 8.2.2
- **Libvirt**: 10.0.0
- **Conda**: For environment management (recommended)

# 🚀 Set up
### 1. Clone the Repository
```shell
git clone https://github.com/SEC-bench/SEC-bench.git
cd pkus
```

### 2. Configure a VM with AMD-SEV support
- Create a qcow2-format virtual disk and modify the `disk` option in `scripts/create_vm.sh`
- Prepare the VM image and modify the `location` option in `scripts/create_vm.sh`

```shell
bash scripts/create_vm.sh
```

### 3. Start VM
Modify `hda` option and `drive` option in `scripts/start_vm.sh` and then run
```shell
bash scripts/start_vm.sh
```

### 4. Run the main program
```shell
python client.py --role=host
python client.py --role=guest
```


# 📣 Precautions
1. You can define the trace by changing the options at the beginning of client.py
2. Remember modifing the model path in client.py before running
3. When unexpected memory leak happens if you hack the code, use `scripts/clear_ivshmem.sh` to clear the IVSHMEM
4. You can use `scripts/update_code.sh` to easily synchronous your code in TEE.


