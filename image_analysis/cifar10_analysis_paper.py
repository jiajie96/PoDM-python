import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import torchvision.utils as tvu


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_10 = unpickle('image_analysis/cifar-10/cifar-10-batches-py/data_batch_1')
print(cifar_10.keys())

X_train = cifar_10[b'data']
print(X_train.shape)

X_train_r = X_train[:,:1024].reshape((10000,32,32))
X_train_g = X_train[:, 1024:2048].reshape((10000,32,32))
X_train_b = X_train[:, 2048:].reshape((10000,32,32))

cifar_10_images = np.stack([X_train_r,X_train_g,X_train_b],3)


cifar_10_images = cifar_10_images/127.5-1
print(cifar_10_images.shape)
print(cifar_10_images.min())
print(cifar_10_images.max())
 
t_steps = [0, 0.02, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]

init_checking_images = torch.tensor(cifar_10_images[:100])

list_of_flatten = [[init_checking_images[:,:,:,0].flatten()],[init_checking_images[:,:,:,1].flatten()],[init_checking_images[:,:,:,2].flatten()]]
list_of_images = [init_checking_images[5]]
print(np.asarray(list_of_images[0]).std())

for i, t_cur in enumerate(t_steps[1:]):
    for k, _color in zip(range(3), ['r','g','b']):
        checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
        list_of_flatten[k].append(checking_images[:,:,:,k].flatten())
    
    list_of_images.append(checking_images[5])
# figsize=(10, 10)
num_bins = 100
plt.figure(figsize=(56, 4))
plt.ioff()
for t in range(len(t_steps)):
    plt.subplot(2, len(t_steps), t+1)
    for k, _color in zip(range(3), ['r','g','b']):
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.hist(list_of_flatten[k][t].numpy(), num_bins, density=True, color= _color, label = _color, alpha=0.3)
        title = r'$\sigma=$'+ f'%.2f'%t_steps[t]
        plt.title(title)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.yticks([]) 
        plt.xticks([-1,1],['-1','1']) 
        # plt.axis('off')
    plt.subplot(2, len(t_steps), t+len(t_steps)+1)
    plt.imshow((list_of_images[t]/2+0.5).numpy())
    plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0.2)

plt.show()
    # plt.title('%.3f' %t_cur.cpu())