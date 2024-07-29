import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats



linestyle_str = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
    ('solid',                 (0, ())),
    ('dotted',                (0, (1, 1))),
    ('dashed',                (0, (5, 5))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('densely dotted',        (0, (1, 1))),
    ('loosely dotted',        (0, (1, 10))),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed',        (0, (5, 10))),
    ('densely dashed',        (0, (5, 1))),
    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

num_steps = 1000000
# step_indices = torch.arange(num_steps)
# sigma_max = 80
# sigma_min = 0.002
# rho_index = 7
# sigmas = (sigma_max ** (1 / rho_index) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho_index) - sigma_max ** (1 / rho_index))) ** rho_index
# plt.plot(step_indices+1, sigmas, color='black')
# plt.xticks(np.arange(0,18,1))
# plt.show()

# plt.figure(figsize=(12, 12))
# for i, rho_index in enumerate([7,5,3,1]):
#     sigmas = (sigma_max ** (1 / rho_index) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho_index) - sigma_max ** (1 / rho_index))) ** rho_index
#     (name, linestyle)  = linestyle_tuple[i]
#     label = "Sampling with rho " + str(rho_index)
#     plt.plot(step_indices+1, sigmas, label = label, color='black', linestyle=linestyle)

# p_mean = -1.2
# p_std = 1.2
# sigmas_training = (torch.randn(num_steps) * p_std + p_mean).exp()
# num_bins = 10000
# # the histogram of the data
# n, bins, patches = plt.hist(sigmas_training, num_bins, density=True, color= "#A9A9A9", label = 'EDM', alpha = 0.5)

# p_mean = 0.5
# p_std = 1.2
# sigmas_training = (torch.randn(num_steps) * p_std + p_mean).exp()
# num_bins = 10000
# # the histogram of the data
# n, bins, patches = plt.hist(sigmas_training, num_bins, density=True, color= 'r')

sigma_start = 5.3
sigma_end = 0.6
  
p_mean = math.log(sigma_end*sigma_start)/2
print(p_mean)
p_std = math.log(sigma_start/sigma_end)/2

x = np.linspace(p_mean - 3*p_std, p_mean + 3*p_std, 100)
normal_curve = stats.norm.pdf(x, p_mean, p_std)

markers_on = [p_mean - 3 * p_std, p_mean - p_std, p_mean + p_std, p_mean + 3 * p_std]
# plt.plot(xs, ys, '-gD', markevery=markers_on
plt.figure(figsize=(3, 3))
plt.plot(x, normal_curve)

plt.xticks(markers_on, [r'$ln(\sigma_{min}$)', r'$ln(\sigma_{end}$)', r'$ln(\sigma_{start}$)', r'$ln(\sigma_{max}$)'])
plt.vlines(markers_on, 0, 0.4, colors='r', linestyles='dashed')

# plt.axvline(x=0.22058956)
# plt.ylim(0, 1)

# plt.yticks(np.arange(0,81,5))
# plt.legend()
plt.xlabel(r'$ln(\sigma$)')
plt.ylabel('Frequence')
# plt.savefig('sigmas_p_mean.svg', bbox_inches='tight')
plt.show()