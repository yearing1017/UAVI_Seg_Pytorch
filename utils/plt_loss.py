import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
 
'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2])) 
        x.append((row[1]))
    return x ,y
 
 
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
 
 
plt.figure()
x2,y2=readcsv("acnet_1014_loss/train_loss.csv")
plt.plot(x2, y2, color='blue', label='train_loss')
#plt.plot(x2, y2, '.', color='red')
 
x,y=readcsv("acnet_1014_loss/val_loss.csv")
plt.plot(x, y, 'g',label='val_loss')

 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
 
plt.ylim(0, 16)
plt.xlim(0, 104800)
plt.xlabel('Steps',fontsize=20)
plt.ylabel('Score',fontsize=20)
plt.legend(fontsize=16)
plt.show()