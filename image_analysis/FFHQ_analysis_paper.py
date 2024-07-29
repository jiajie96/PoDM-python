import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import torchvision.utils as tvu
import cv2
from scipy.stats import shapiro

def KL(P,Q):
    var_p = P.var()
    var_q = Q.var()
    mean_p = P.mean()
    mean_q = Q.mean()

    divergence = np.log(var_q.sqrt()/var_p.sqrt()) + (var_p + (mean_p - mean_q)**2)/(2 * var_q) - 0.5
     
    return divergence

dataset_path = 'FFHQ'
dataset_files_list = os.listdir(dataset_path)
print(len(dataset_files_list))

row_images=[]
for dataset_file_name in dataset_files_list[:5000]:
        image=cv2.imread(os.path.join(dataset_path, dataset_file_name), cv2.IMREAD_COLOR)   
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        row_images.append(image)
        
row_images = np.array(row_images)
row_images = row_images.astype('int')

X_train_r = row_images[:,:,:,2]
X_train_g = row_images[:,:,:,1]
X_train_b = row_images[:,:,:,0]
row_images = np.stack([X_train_r,X_train_g,X_train_b],3)
np.random.shuffle(row_images)
print(row_images.shape)
FFHQ_images = row_images/255
row_images = np.double(row_images)
# plt.imshow((row_images[0]))
print(FFHQ_images[:,:,:,0].min())
print(FFHQ_images[:,:,:,0].max())
np.save('FFHQ_1.npy', FFHQ_images)



# # t_steps = [0, 0.02, 0.2,0.25,0.3,0.35, 0.4,0.5, 0.6, 1, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6, 7, 8, 9, 10, 11, 12]
# t_steps = [0, 0.02, 0.2, 0.6, 1, 3.0, 6.0, 10]

# init_checking_images = torch.tensor(FFHQ_images[:100])

# list_of_flatten = [[init_checking_images[:,:,:,0].flatten()],[init_checking_images[:,:,:,1].flatten()],[init_checking_images[:,:,:,2].flatten()]]
# list_of_images = [init_checking_images[3]]
# list_of_kld = [[KL(init_checking_images[:,:,:,0].flatten(),init_checking_images[:,:,:,1].flatten()),KL(init_checking_images[:,:,:,0].flatten(),init_checking_images[:,:,:,2].flatten()),KL(init_checking_images[:,:,:,1].flatten(),init_checking_images[:,:,:,2].flatten())]]
# list_of_shapiro = [[shapiro(init_checking_images[:,:,:,0].flatten()).pvalue,shapiro(init_checking_images[:,:,:,1].flatten()).pvalue,shapiro(init_checking_images[:,:,:,2].flatten()).pvalue]]
# print(np.asarray(list_of_images[0]).std())

# for i, t_cur in enumerate(t_steps[1:]):
#     for k, _color in zip(range(3), ['r','g','b']):
#         checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
#         list_of_flatten[k].append(checking_images[:,:,:,k].flatten())
#     # list_of_shapiro.append([shapiro(checking_images[:,:,:,0].flatten()).pvalue,shapiro(checking_images[:,:,:,1].flatten()).pvalue,shapiro(checking_images[:,:,:,2].flatten()).pvalue])
#     # list_of_kld.append([KL(checking_images[:,:,:,0].flatten(),checking_images[:,:,:,1].flatten()),KL(checking_images[:,:,:,0].flatten(),checking_images[:,:,:,2].flatten()),KL(checking_images[:,:,:,1].flatten(),checking_images[:,:,:,2].flatten())])

#     # if KL(checking_images[:,:,:,0].flatten(),checking_images[:,:,:,1].flatten())<=0.02 and KL(checking_images[:,:,:,1].flatten(),checking_images[:,:,:,2].flatten())<=0.02 and KL(checking_images[:,:,:,0].flatten(),checking_images[:,:,:,2].flatten())<=0.02:
#         # print('here:' ,t_cur)

#     list_of_images.append(checking_images[3])

# print(list_of_shapiro)
# print(list_of_kld)

# # figsize=(10, 10)
# num_bins = 100
# plt.figure(figsize=(56, 4))
# plt.ioff()
# for t in range(len(t_steps)):
#     plt.subplot(2, len(t_steps), t+1)
#     # for k, _color in zip(range(1), ['b']):
#     for k, _color, _name in zip(range(3), ['r','y','b'], ['r','g','b']):
#         plt.tight_layout(h_pad=0, w_pad=0)
#         # plt.hist(np.stack([list_of_flatten[0][t].numpy().flatten(), list_of_flatten[1][t].numpy().flatten(), list_of_flatten[2][t].numpy().flatten()]).flatten(), num_bins, density=True, color= _color, label = _color, alpha=0.5)
#         plt.hist(list_of_flatten[k][t].numpy(), num_bins, density=True, color= _color, label = _name, alpha=0.5)
#         title = r'$\sigma=$'+ f'%.2f'%t_steps[t]
#         if t==0:
#              plt.legend(loc='upper right')
#         plt.title(title)
#         plt.yticks([]) 
#         if t<=3:
#             plt.xticks([-1,1],['-1','1']) 
#         # plt.xlabel('pixel value')
#         # plt.axis('off')
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)

#     plt.subplot(2, len(t_steps), t+len(t_steps)+1)
#     plt.imshow((list_of_images[t]/2+0.5).numpy())
#     plt.axis('off')
#     plt.tight_layout(h_pad=0, w_pad=0)

# plt.show()
#     # plt.title('%.3f' %t_cur.cpu())