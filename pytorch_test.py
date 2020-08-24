import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


x = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(1, 3, 2)
x = Variable(x)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step
print(x)