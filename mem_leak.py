import numpy as np
import gc
import resource
import torch
import torch.nn as nn
from torch.autograd import Variable

model = nn.Linear(100, 2)

i = 0
while True:
    i += 1

    np_arr = np.ones(100)
    tensor = torch.from_numpy(np_arr[None]).float() # Issue seems to be use of [None] or [np.newaxis]
    variable = Variable(tensor)
    output = model(variable)

    if i % 1000 == 0:
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("{:.2f} MB".format(max_mem_used / 1024))
