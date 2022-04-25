
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import csv

focus_val = "focused"

path = rf'C:\Users\maham\Documents\Senior_Design\tune-in\data\{focus_val}\csv\new_1' # use your path
all_files = glob.glob(path + "/*.csv")

conc_array = []
i = 0
max_beta = 0;
int_beta = 0;
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # get average of all beta waves for properly fitting electrodes
    # 4 beta readings
    #beta0 = df.loc[:, "beta0"]
    #beta1 = df.loc[:, "beta1"]
    #beta2 = df.loc[:, "beta2"]
    #beta3 = df.loc[:, "beta3"]
    #avg_beta = (beta0 + beta1 + beta2 + beta3)/4
    #avg_beta = np.fft.fft(avg_beta)
    # get focused state
    #conc_col = df.loc[:,"concentration"]
    #conc_val = float(conc_col[1])
    #conc_array.append(conc_val)
    avg_beta = df.loc[:, "beta"]
    int_beta = max(avg_beta)
    if(int_beta > max_beta):
        max_beta = int_beta
    # make histogram
    #plt.plot(dtft)
    plt.hist(avg_beta)
    #plt.imshow(avg_beta, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.ylim([0, 20])
    char = chr(i)
    savefig_name = rf'C:\Users\maham\Documents\Senior_Design\tune-in\data\focused'
    temp2 = '.png'
    plt.savefig(savefig_name + "\\" + "0420" + str(i) + ".png")
    plt.close()
    i += 1
    beta0 = 0
    beta1 = 0
    beta2 = 0
    beta3 = 0

print(max_beta)
#conc_df = pd.DataFrame(conc_array)
#conc_df.to_csv('concentration_values.csv',index=False,header=False)


