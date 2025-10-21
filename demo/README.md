# demo代码使用方法

1. 将`model`目录及相应模型文件置于`demo`目录下，修改`client.py`中`BASE_MODEL_DIR` `LORA_MODEL_DIR`
2. 使用`scripts`中的脚本创建好并进入虚拟机后，运行`update_code.sh`将代码上传到虚拟机
3. 在虚拟机中运行`python client.py --role=guest`，在物理机上运行`python client.py --role=host`