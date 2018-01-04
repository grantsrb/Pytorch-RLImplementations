import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    log_file = str(sys.argv[1])        
else:
    print("Need to specify log file")

log = open(log_file, 'r')
lines = []
for line in log:
    lines.append(line)

data = []
T = []

for i,line in enumerate(lines[2:]):
    if "Batch" not in line and "T" not in line and i > 87:
        temp = [str(x) for x in line.split(",")]
        data.append([float(x) for x in temp[1:]])
        T.append(int(temp[0].strip()))
    else:
        failure = True

data = np.asarray(data).astype(np.float32)
T = np.asarray(T).astype(np.int32)

for i, field in enumerate(lines[1].split(",")[1:]):
    plt.figure(i+1)
    plt.plot(T, data[:,i])
    plt.xlabel("T")
    plt.ylabel(field.strip())
    plt.savefig(field.strip()+"_"+log_file[:-4]+".png")

   
   
