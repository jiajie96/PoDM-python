import torch
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import yaml
import argparse
import copy
import torchvision.utils as tvu
import shutil
import glob
import time
import torch.utils.data as data 
import torch.optim as optim
import scipy.stats as stats
import math
from scipy.stats import shapiro
import scipy

# plt.style.use('_mpl-gallery')

def KS_2S(P, Q):
    KS = scipy.stats.anderson_ksamp([P, Q])

    return KS.statistic
                     

val_images = np.load('image_analysis/Biked/biked_val_256.npy', mmap_mode='r')
val_images = val_images/0.5-1


t_steps = np.linspace(0.5, 80, 1000)

list_of_list_of_KS = []

for line in range(2):
    init_checking_images = torch.tensor(val_images[line:line+1])

    checking_image_values = init_checking_images.flatten()
    feature_map = []
    background_map = []
    for i in range(checking_image_values.shape[0]):
        if checking_image_values[i]<=0.85:
            feature_map.append(i)
        elif checking_image_values[i]>0.85:
            background_map.append(i)
            
    list_of_KS = []

    for i, t_cur in enumerate(t_steps[0:]):
        checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
        list_of_KS.append(KS_2S(checking_images.flatten()[feature_map], checking_images.flatten()[background_map]))

    list_of_KS = np.asarray(list_of_KS)


    list_of_list_of_KS.append(list_of_KS)

list_of_list_of_KS = np.asarray(list_of_list_of_KS)

max_list_of_KS = [list_of_list_of_KS[:,k].max() for k in range(len(t_steps))]
mean_list_of_KS = [list_of_list_of_KS[:,k].mean() for k in range(len(t_steps))]
min_list_of_KS = [list_of_list_of_KS[:,k].min() for k in range(len(t_steps))]


y = [0.02, 0.2, 0.5, 1]
x = [0.5, 1.5, 3.0, 5.0, 10.0, 20, 50, 80]
plt.figure(figsize=(5, 4))

y_scale = 1
plt.fill_between(np.asarray(t_steps)**(1/3), np.asarray(min_list_of_KS)**(1/y_scale), np.asarray(max_list_of_KS)**(1/y_scale), alpha=.5, linewidth=0)
plt.plot(np.asarray(t_steps)**(1/3), np.asarray(mean_list_of_KS)**(1/y_scale), linewidth=2)

plt.xticks(np.asarray(x)**(1/3), x)
plt.yticks(np.asarray(y)**(1/y_scale), y)
plt.xlabel(r'$\sigma$')
plt.title('The k-sample Anderson-Darling test')
plt.ylabel('result')
plt.show()
