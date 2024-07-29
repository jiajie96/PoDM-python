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

def anderson(P):
    likelyhood = scipy.stats.anderson(P, 'norm')

    return likelyhood.statistic
                     

val_images = np.load('image_analysis/Biked/biked_val_256.npy', mmap_mode='r')
val_images = val_images/0.5-1


t_steps = np.linspace(0, 10, 100)

list_of_list = []

for line in range(100):
    init_checking_images = torch.tensor(val_images[line:line+1])

    checking_image_values = init_checking_images.flatten()
    feature_map = []
    background_map = []
    for i in range(checking_image_values.shape[0]):
        if checking_image_values[i]<=0.85:
            feature_map.append(i)
        elif checking_image_values[i]>0.85:
            background_map.append(i)
            
    list = []

    for i, t_cur in enumerate(t_steps[0:]):
        checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
        list.append(anderson((checking_images.numpy()).flatten()[feature_map]))

    list = np.asarray(list)


    list_of_list.append(list)

list_of_list = np.asarray(list_of_list)

max_list = [list_of_list[:,k].max() for k in range(len(t_steps))]
mean_list = [list_of_list[:,k].mean() for k in range(len(t_steps))]
min_list = [list_of_list[:,k].min() for k in range(len(t_steps))]


y = [0.02, 0.2, 0.5, 1]
x = np.linspace(0, 10, 11)
plt.figure(figsize=(5, 4))

y_scale = 11
plt.fill_between(np.asarray(t_steps) , np.asarray(min_list)**(1/y_scale), np.asarray(max_list)**(1/y_scale), alpha=.5, linewidth=0)
plt.plot(np.asarray(t_steps) , np.asarray(mean_list)**(1/y_scale), linewidth=2)

plt.xticks(np.asarray(x) , x)
plt.yticks(np.asarray(y)**(1/y_scale), y)
plt.xlabel(r'$\sigma$')
plt.title('Kolmogorov-Smirnov')
plt.ylabel('KS')
plt.show()
