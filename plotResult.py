import matplotlib.pyplot as plt
import numpy as np

data1 = [[36, 20, 20, 36, 43, 33, 9, 30, 33, 19],[10, 23, 24, 17, 24, 28, 42, 36, 36, 42],[41, 53, 56, 44, 60, 57, 58, 41, 25, 55]]
data2 = [[15, 13, 15, 37, 15, 24, 15, 20, 22, 28],[38, 26, 28, 24, 22, 14, 21, 39, 35, 9],[26, 41, 29, 40, 34, 43, 48, 41, 37, 43]]
data3 = [[19, 9, 17, 9, 11, 8, 28, 17, 14, 20],[37, 16, 21, 13, 12, 18, 19, 31, 22, 22],[22, 31, 34, 28, 36, 31, 24, 17, 29, 31]]

all_data = [data1,data2,data3]
environments = ["400x400","320x320","240x240"]
labels = ['DQL-11', 'DQL-23', 'A*']

for environment,data in zip(environments,all_data):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    bplot1 = ax1.boxplot(data,
	                     vert=True,
	                     patch_artist=True,  
	                     labels=labels)  
    colors = ['cyan', 'orange', 'lightgreen']
    for bplot in ([bplot1]):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    ax1.set_title('%s environment' %environment)
    ax1.yaxis.grid(True)
    ax1.set_xlabel('Agents')
    ax1.set_ylabel('Food Collected / Score')
    plt.show()
