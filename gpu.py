# 运行以下代码获取对应硬件情况
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
print("Tensor:\n",tensor)
