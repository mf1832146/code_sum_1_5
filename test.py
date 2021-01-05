import torch.nn as nn
import torch
import torch.nn.functional as F

print(F.softmax(torch.Tensor([float('-inf'), 0]), -1))