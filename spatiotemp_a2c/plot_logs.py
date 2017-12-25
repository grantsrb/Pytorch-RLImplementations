import matplotlib.pyplot as plt
import numpy as np
import sys

file_path = str(sys.argv[1])
log = open(file_path, 'r')

lines = []
for line in log:
    lines.append(line)
print(lines[1])
plot_type = str(input())

plot_dict = {k.strip():v for v,k in enumerate(lines[1].split(","))}
print(plot_dict)

data = []
for line in lines[2:]:
    data.append([float(x) for x in line.split(",")])

data = np.asarray(data)

plt.plot(data[:,0], data[:,plot_dict[plot_type]])
plt.show()
