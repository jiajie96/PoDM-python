import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import torchvision.utils as tvu
import cv2

dataset_path = 'image_analysis/FFHQ_analysis/FFHQ'
dataset_files_list = os.listdir(dataset_path)
print(len(dataset_files_list))

row_images=[]
for dataset_file_name in dataset_files_list[:100]:
        image=cv2.imread(os.path.join(dataset_path, dataset_file_name), cv2.IMREAD_COLOR)
        row_images.append(image)
        
row_images = np.array(row_images)
row_images = row_images.astype('int')


plt.figure(figsize=(40, 200))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(row_images[i, :, :, :])
    plt.axis('off')
plt.show()

print(row_images.shape)
FFHQ_images = row_images/127.5-1
print(FFHQ_images[:,:,:,0].min())
print(FFHQ_images[:,:,:,0].max())

num_steps = 50
sigma_max = 10
sigma_min = 0.02
step_indices = torch.arange(num_steps)
t_steps = sigma_max + step_indices / (num_steps - 1) * (sigma_min- sigma_max )
# t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
t_steps = t_steps.flip(0)


init_checking_images = torch.tensor(FFHQ_images[:100])

for i, t_cur in enumerate( t_steps[:]):
    plt.figure(figsize=(10, 10))
    for k, _color in zip(range(3), ['r','g','b']):
        checking_images = init_checking_images + (t_cur.cpu()) * torch.randn_like(init_checking_images)
        checking_image_values = checking_images[:,:,:,k].flatten()
        num_bins = 100
        # plt.hist(checking_image_values[feature_map].numpy(), num_bins, density=True, color= 'r',label = 'feature', alpha=0.5)
        # plt.hist(checking_image_values[background_map].numpy(), num_bins, density=True, color= 'b',label = 'background', alpha=0.5)
        plt.hist(checking_image_values.numpy(), num_bins, density=True, color= _color,label = _color, alpha=0.3)

    plt.title('%.3f' %t_cur.cpu())
    plt.savefig('image_analysis/FFHQ_analysis/per_pixel_value_plot/noise_scale+At_%.3f.png' %t_cur.cpu(), bbox_inches='tight')

    file_name = f'noise_scale_sigma_%.3f.png' %t_cur
#     tvu.save_image(checking_images, os.path.join(f'checking_sigma_start_and_end(straight_forward)', file_name))
    print(f'noise scale: %.3f'%t_cur + f', var: %.3f'%torch.var(checking_images) +f', mean: %.3f'%torch.mean(checking_images) )