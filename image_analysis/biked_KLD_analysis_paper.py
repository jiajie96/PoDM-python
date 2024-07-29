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
import matplotlib 
import torch.optim as optim
import scipy.stats as stats
import math
from scipy.stats import shapiro

# plt.style.use('_mpl-gallery')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

def KL(P,Q):
    var_p = P.var()
    var_q = Q.var()
    mean_p = P.mean()
    mean_q = Q.mean()

    divergence = np.log(var_q.sqrt()/var_p.sqrt()) + (var_p + (mean_p - mean_q)**2)/(2 * var_q) - 0.5
     
    return divergence

val_images = np.load('image_analysis/Biked/biked_val_256.npy', mmap_mode='r')
val_images = val_images/0.5-1


t_steps = np.linspace(0.5, 20, 1000)

list_of_list_of_kld = []
# list_of_list_of_shapiro = []

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
            
    list_of_kld = []
    # list_of_shapiro = [shapiro(init_checking_images.flatten()[feature_map]).statistic]

    for i, t_cur in enumerate(t_steps[0:]):
        checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
        list_of_kld.append(KL(checking_images.flatten()[feature_map], checking_images.flatten()[background_map]))
        # list_of_shapiro.append(shapiro(checking_images.flatten()[feature_map]).statistic) 

    list_of_kld = np.asarray(list_of_kld)
    # list_of_shapiro = np.asarray(list_of_shapiro)


    list_of_list_of_kld.append(list_of_kld)
    # list_of_list_of_shapiro.append(list_of_shapiro)

list_of_list_of_kld = np.asarray(list_of_list_of_kld)
# list_of_list_of_shapiro = np.asarray(list_of_list_of_shapiro)

# print(list_of_list_of_shapiro)
max_list_of_kld = [list_of_list_of_kld[:,k].max() for k in range(len(t_steps))]
mean_list_of_kld = [list_of_list_of_kld[:,k].mean() for k in range(len(t_steps))]
min_list_of_kld = [list_of_list_of_kld[:,k].min() for k in range(len(t_steps))]

# max_list_of_shapiro = [list_of_list_of_shapiro[:,k].max() for k in range(len(t_steps))]
# mean_list_of_shapiro = [list_of_list_of_shapiro[:,k].mean() for k in range(len(t_steps))]
# min_list_of_shapirod = [list_of_list_of_shapiro[:,k].min() for k in range(len(t_steps))]

y = [0.02, 0.2, 0.5, 1, 2, 5 ]
x = [0.5, 1.5, 3.0, 5.0, 10.0, 20]
plt.figure(figsize=(5, 4))

y_scale = 1.5
plt.fill_between(np.asarray(t_steps)**(1/3), np.asarray(min_list_of_kld)**(1/y_scale), np.asarray(max_list_of_kld)**(1/y_scale), alpha=.5, linewidth=0)
plt.plot(np.asarray(t_steps)**(1/3), np.asarray(mean_list_of_kld)**(1/y_scale), linewidth=2)

# plt.plot(np.asarray(t_steps)**(1/3), np.asarray(list_of_kld)**(1/3.5))
plt.xticks(np.asarray(x)**(1/3), x)
plt.yticks(np.asarray(y)**(1/y_scale), y)
# plt.title('KLD between object and background pixel values')
plt.xlabel(r'$\sigma$')
plt.ylabel('KLD')
plt.show()
plt.savefig("KLD_BO.png")