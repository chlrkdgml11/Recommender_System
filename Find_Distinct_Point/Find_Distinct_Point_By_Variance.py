import numpy as np

data = np.loadtxt("./Find_Distinct_Point/rmse.txt", delimiter = ",", dtype = np.float64)

data = np.array(data)

rmse = []

for i in range(20):
    rmse.append(data[i][1])

rmse = np.array(rmse)
print(rmse)

group = []

for i in range(16):
    group.append(np.var(rmse[i:i+5]))
    print(round(np.var(rmse[i:i+5]), 8))


