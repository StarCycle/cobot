import torch

ckpt_path = "/path/to/your/checkpoint.pth"  # 修改为你训练生成的 ckpt 路径
state_dict = torch.load(ckpt_path, map_location="cpu")  # 只加载到 CPU，避免显存占用

# 如果是完整 checkpoint 包含其它字段，找到 state_dict
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

# 打印所有键
for k in state_dict.keys():
    print(k)
