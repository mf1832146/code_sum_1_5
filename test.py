import torch

print(torch.log(torch.Tensor([1e-100 + 1e-9])))