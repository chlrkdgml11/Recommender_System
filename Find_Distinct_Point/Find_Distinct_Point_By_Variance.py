import numpy as np
import matplotlib.pyplot as plt


over_20_data_users = np.loadtxt("./Find_Distinct_Point/over_20_data_users.txt", delimiter = ",", dtype = np.float64)






for x in over_20_data_users:


    data = np.loadtxt("./Find_Variance_Text/rmse_%d.txt" %int(x[0]), delimiter = "\t", dtype = np.float64)

    data = np.array(data)

    rmse = []

    for i in range(len(data)):
        rmse.append(data[i][1])

    rmse = np.array(rmse)

    group = []

    plt.figure()
    plt.title('%d User' %int(x[0]))
    plt.xlabel('Group')
    plt.ylabel('Variance')
    plt.axis([0, 17, -0.05, 1.5])
    plt.xticks(list(range(16))[::3])
    for i in range(16):
        var = round(np.var(rmse[i:i+3]), 6)
        group.append(var)
        plt.plot(i+1, var, 'ro')

    # plt.show()
    plt.savefig('./Find_Variance_Image/%d.png' %int(x[0]))
