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

plt.figure(figsize=(40, 200))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(cifar_10_images[i, :, :, :])
    plt.axis('off')
plt.show()

cifar_10_images = cifar_10_images/127.5-1
print(cifar_10_images.shape)
print(cifar_10_images.min())
print(cifar_10_images.max())



num_steps = 18
sigma_max = 80
sigma_min = 0.002
rho = 5
# Time step discretization.
step_indices = torch.arange(num_steps)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
t_steps = t_steps.flip(0)
print(t_steps)

cifar_10_images_tensor = torch.tensor(cifar_10_images).float()
cifar_10_images_tensor = cifar_10_images_tensor.permute((0, 3, 1, 2))

init_checking_images = cifar_10_images_tensor[:20]
init_var = torch.var(init_checking_images)

init_mean = torch.mean(init_checking_images)
print('init var: %.2f' %init_var, 'init mean: %.2f' %init_mean)
for i, t_cur in enumerate( t_steps[:]):
    checking_images = init_checking_images + (t_cur.cpu()) * torch.randn_like(init_checking_images)
    checking_image_values = checking_images.flatten()
    
    num_bins = 10
    plt.hist(checking_image_values.numpy(), num_bins, density=True, color= 'r')
    plt.savefig('image_analysis/cifar-10/per_pixel_value_plot/noise_scale+At_%.3f.png' %t_cur.cpu(), bbox_inches='tight')

    file_name = f'noise_scale_sigma_%.3f.png' %t_cur
    tvu.save_image(checking_images[:5], os.path.join(f'image_analysis/cifar-10/noised_images', file_name))
    # normalized_var = torch.var(checking_images)/(t_cur.cpu()**2)
    # normalized_mean = torch.mean(checking_images)-init_mean
    print(f'noise scale: %.3f'%t_cur + f', var: %.3f'%torch.var(checking_images) + f', mean: %.3f'%torch.mean(checking_images))

# init_var_per_pixel = torch.zeros((1,256,256))
# init_mean_per_pixel = torch.zeros((1,256,256))
# for i in range(256):
#     for j in range(256):
#         init_var_per_pixel[:,i,j] = torch.var(init_checking_images[:,:,i,j])
#         init_mean_per_pixel[:,i,j] = torch.mean(init_checking_images[:,:,i,j])
# tvu.save_image(init_var_per_pixel, os.path.join(f'single_pixel_values_plot(straight_forward)/init_var_per_pixel.png'))
# tvu.save_image(init_mean_per_pixel, os.path.join(f'single_pixel_values_plot(straight_forward)/init_mean_per_pixel.png'))

for i, t_cur in enumerate( t_steps[:]):
    checking_images = init_checking_images + (t_cur.cpu()) * torch.randn_like(init_checking_images)
    var_per_pixel = torch.zeros((1,32,32))
    mean_per_pixel = torch.zeros((1,32,32))
    for i in range(32):
        for j in range(32):
            var_per_pixel[:,i,j] = torch.var(checking_images[:,:,i,j])
            mean_per_pixel[:,i,j] = torch.mean(checking_images[:,:,i,j])
            
    
#     tvu.save_image(var_per_pixel, os.path.join(f'single_pixel_values_img(straight_forward)/var_per_pixel_%.3f.png' %t_cur))
#     tvu.save_image(mean_per_pixel, os.path.join(f'single_pixel_values_img(straight_forward)/mean_per_pixel_%.3f.png' %t_cur))
    var_per_pixel = var_per_pixel.flatten()
    mean_per_pixel = mean_per_pixel.flatten()
    var_of_mean = torch.var(mean_per_pixel)
    print(f'noise scale: %.3f'%t_cur + f', var of mean: %.3f'%torch.var(var_of_mean))
    x = np.arange(32*32)
    
    plt.figure()
    plt.plot(x, var_per_pixel, color= 'r', label='variance', alpha=0.5)
    plt.plot(x, mean_per_pixel, color= 'b', label='mean', alpha=0.5)
    plt.legend()
    plt.savefig('image_analysis/cifar-10/single_pixel_values_plot(straight_forward)/statistic_per_pixel_%.3f.png' %t_cur.cpu(), bbox_inches='tight')
    

