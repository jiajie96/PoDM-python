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
import cv2

def KL(P,Q):
    var_p = P.var()
    var_q = Q.var()
    mean_p = P.mean()
    mean_q = Q.mean()

    divergence = np.log(var_q.sqrt()/var_p.sqrt()) + (var_p + (mean_p - mean_q)**2)/(2 * var_q) - 0.5
     
    return divergence

shoe_dataset_path = 'image_analysis/QMUL-Shoe-Chair/ShoeV2/trainB'
shoe_dataset_files_list = os.listdir(shoe_dataset_path)
print(len(shoe_dataset_files_list))

row_images=[]
for dataset_file_name in shoe_dataset_files_list[:]:
        image=cv2.imread(os.path.join(shoe_dataset_path, dataset_file_name), cv2.IMREAD_COLOR) 
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        row_images.append(image)
        
row_images = np.array(row_images)
row_images = row_images.astype('int')

X_train_r = row_images[:,:,:,0]
X_train_g = row_images[:,:,:,1]
X_train_b = row_images[:,:,:,2]
row_images = np.stack([X_train_r,X_train_g,X_train_b],3)
print(row_images.shape)
saving_iamges = row_images/255
row_images = row_images/127.5-1
QMUL_images = row_images.astype('float32')
saving_iamges = saving_iamges.astype('float32')
ival = saving_iamges[:100,:,:,:]
itrain = saving_iamges[100:,:,:,:]

print(itrain[:,:,:,:].min())
print(itrain[:,:,:,:].max())

np.save('./Shoe_train_256.npy', itrain.astype('float32'))
np.save('./Shoe_val_256.npy', ival.astype('float32'))

print(QMUL_images[:,:,:,0].min())
print(QMUL_images[:,:,:,0].max())

"""color without object background seperating"""
# t_steps = [0, 0.02, 0.2, 0.4, 0.6, 1, 5.0, 10.0]

# init_checking_images = torch.tensor(QMUL_images[:100])

# list_of_flatten = [[init_checking_images[:,:,:,0].flatten()],[init_checking_images[:,:,:,1].flatten()],[init_checking_images[:,:,:,2].flatten()]]
# list_of_images = [init_checking_images[3]]

# for i, t_cur in enumerate(t_steps[1:]):
#     for k, _color in zip(range(3), ['r','g','b']):
#         checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
#         list_of_flatten[k].append(checking_images[:,:,:,k].flatten())
#     list_of_images.append(checking_images[3])

# # figsize=(10, 10)
# num_bins = 100
# plt.figure(figsize=(56, 4))
# plt.ioff()
# for t in range(len(t_steps)):
#     plt.subplot(2, len(t_steps), t+1)
#     for k, _color in zip(range(3), ['r','g','b']):
#         plt.tight_layout(h_pad=0, w_pad=0)
#         plt.hist(list_of_flatten[k][t].numpy(), num_bins, density=True, color= _color, label = _color, alpha=0.3)
#         title = r'$\sigma=$'+ f'%.2f'%t_steps[t]
#         plt.title(title)
#         plt.yticks([]) 
#         plt.xticks([]) 
#         plt.xlabel('pixel value')
#         # plt.axis('off')
#     plt.subplot(2, len(t_steps), t+len(t_steps)+1)
#     plt.imshow(list_of_images[t].numpy())
#     plt.axis('off')
#     plt.tight_layout(h_pad=0, w_pad=0.2)

# plt.show()
    # plt.title('%.3f' %t_cur.cpu())


# t_steps = [0, 0.02,0.1, 0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1.0,1.5,2,2.5, 3.0, 6.0,7,8,9, 10.0]
# # t_steps = [0, 0.6, 5.3, 40]
# # t_steps = [0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
# # t_steps = [0]
# # 
# init_checking_images = torch.tensor(QMUL_images[:100])

# checking_image_values = init_checking_images.flatten()
# feature_map = []
# background_map = []
# for i in range(checking_image_values.shape[0]):
#     if checking_image_values[i]<=0.85:
#         feature_map.append(i)
#     elif checking_image_values[i]>0.85:
#         background_map.append(i)
        
# list_of_flatten = [init_checking_images.flatten()]
# list_of_feature = [init_checking_images.flatten()[feature_map]]
# print(np.asarray(list_of_feature[0]).std())

# list_of_background = [init_checking_images.flatten()[background_map]]
# list_of_images = [init_checking_images[0]]
# list_of_kld = [KL(init_checking_images.flatten()[feature_map], init_checking_images.flatten()[background_map])]
# list_of_shapiro = [shapiro(init_checking_images.flatten()[feature_map]).statistic]

# for i, t_cur in enumerate(t_steps[1:]):
#     checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
#     list_of_flatten.append(checking_images.flatten())
#     list_of_feature.append(checking_images.flatten()[feature_map])
#     list_of_background.append(checking_images.flatten()[background_map])
#     list_of_kld.append(KL(checking_images.flatten()[feature_map], checking_images.flatten()[background_map]))
#     list_of_images.append(checking_images[0])        
#     list_of_shapiro.append(shapiro(checking_images.flatten()[feature_map])) 

# list_of_shapiro = np.asarray(list_of_shapiro)

# print(list_of_shapiro)
# print(list_of_shapiro.shape)
# print(list_of_kld)
# num_bins = 1000

# o = 0
# for image in list_of_images:
#     plt.figure(figsize=(6, 6))
#     plt.ioff()
#     o+=1 
#     plt.imshow((image/2+0.5).numpy(), cmap='gray')
#     plt.axis('off')
#     plt.tight_layout(h_pad=0, w_pad=0)
#     plt.savefig(f'image_analysis/Biked/save_step_images/{o}.png')

# plt.figure(figsize=(50, 5))
# plt.ioff()
# for t in range(len(t_steps)):
#     plt.subplot(3, len(t_steps), t+1)

#     density = True
#     plt.hist(list_of_flatten[t].numpy(), num_bins, density = density, color= 'black', alpha=0.3)
#     title = r'$\sigma=$'+ f'%.2f'%t_steps[t]
#     plt.title(title)
#     plt.yticks([]) 
#     if t<=3:
#         plt.xticks([-1,1],['-1','1']) 
#     if t <=1:
#         plt.ylim(0, 1)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     # plt.axis('off')
    
#     plt.subplot(3, len(t_steps), t+len(t_steps)+1)

#     plt.imshow((list_of_images[t]/2+0.5).numpy(), cmap='gray')
#     plt.axis('off')

#     plt.subplot(3, len(t_steps), t+2*len(t_steps)+1)

#     if t ==0:
#         plt.hist(list_of_feature[t].numpy(), num_bins, density = False , color= 'r',label = 'object', alpha=0.5)
#         plt.hist(list_of_background[t].numpy(), 90, density = False, color= 'b',label = 'background', alpha=0.5)
#     elif t ==1:
#         plt.hist(list_of_feature[t].numpy(), num_bins, density = False , color= 'r',label = 'object', alpha=0.5)
#         plt.hist(list_of_background[t].numpy(), 3800, density = False, color= 'b',label = 'background', alpha=0.5)
#     else:
#         plt.hist(list_of_feature[t].numpy(), num_bins, density = True , color= 'r',label = 'object', alpha=0.5)
#         plt.hist(list_of_background[t].numpy(), num_bins, density = True, color= 'b',label = 'background', alpha=0.5)
    
#     # plt.ylabel('density') 
#     plt.yticks([]) 
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)

#     if t <=3:
#         plt.xticks([-1,1],['-1','1']) 

#     if t ==0:
#         plt.legend(loc='upper right')
#     if t <=1:
#         # plt.ylim(0, 3)
#         plt.ylim(0, 10000)
#     plt.tight_layout(h_pad=0, w_pad=0)

# plt.show()