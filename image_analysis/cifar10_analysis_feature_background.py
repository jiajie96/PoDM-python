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

# plt.figure(figsize=(40, 200))
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(cifar_10_images[i, :, :, :])
#     plt.axis('off')
# plt.show()

cifar_10_images = cifar_10_images/127.5-1
print(cifar_10_images.shape)
print(cifar_10_images.min())
print(cifar_10_images.max())

# num_steps = 50
# sigma_max = 10
# sigma_min = 0.02
# step_indices = torch.arange(num_steps)
# t_steps = sigma_max + step_indices / (num_steps - 1) * (sigma_min- sigma_max )
# # t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
# t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
# t_steps = t_steps.flip(0)

t_steps = [0, 0.02, 0.2, 0.4, 0.6, 0.8, 1.0]

init_checking_images = torch.tensor(cifar_10_images[:100])

list_of_flatten = [[init_checking_images[:,:,:,0].flatten()],[init_checking_images[:,:,:,1].flatten()],[init_checking_images[:,:,:,2].flatten()]]
list_of_images = [init_checking_images[0]]

for i, t_cur in enumerate(t_steps[1:]):
    for k, _color in zip(range(3), ['r','g','b']):
        checking_images = init_checking_images + (t_cur) * torch.randn_like(init_checking_images)
        list_of_flatten[k].append(checking_images[:,:,:,k].flatten())
        # plt.hist(checking_image_values.numpy(), num_bins, density=True, color= _color,label = _color, alpha=0.3)

    # plt.title('%.3f' %t_cur.cpu())
    # plt.savefig('image_analysis/cifar-10/per_pixel_value_plot/noise_scale+At_%.3f.png' %t_cur.cpu(), bbox_inches='tight')
    list_of_images.append(checking_images[0])
    # file_name = f'noise_scale_sigma_%.3f.png' %t_cur
#     tvu.save_image(checking_images, os.path.join(f'checking_sigma_start_and_end(straight_forward)', file_name))
    # print(f'noise scale: %.3f'%t_cur + f', var: %.3f'%torch.var(checking_images) +f', mean: %.3f'%torch.mean(checking_images) )

# figsize=(10, 10)
num_bins = 100
plt.figure()
plt.ioff()
for t in range(len(t_steps)):
    plt.subplot(2, len(t_steps), t+1)
    for k, _color in zip(range(3), ['r','g','b']):
        plt.hist(list_of_flatten[k][t].numpy(), num_bins, density=True, color= _color, label = _color, alpha=0.3)
        title = r'$\sigma$ = ' + f'$%.3f'%t_steps[t]
        plt.title(title)
        plt.axis('off')
    plt.subplot(2, len(t_steps), t+len(t_steps)+1)
    plt.imshow(list_of_images[t].numpy())
    title = r'$\sigma$' + ' = ' + f'$%.3f'%t_steps[t]
    plt.title(title)
    plt.axis('off')

plt.show()
    # plt.title('%.3f' %t_cur.cpu())