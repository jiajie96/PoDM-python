import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib 
import math
import scipy.stats as stats

from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

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


sigma_start = 5.3
sigma_end = 0.6

sigma_max = sigma_start**2/sigma_end
sigma_min = sigma_end**2/sigma_start

y_scale = 3

ystick_labels = [0.001, 0.6 , 5.3, 10, 20, 50, 85]

x_scale = 1

# Start with a square Figure.
fig = plt.figure(figsize=(8, 4)) 
gs = fig.add_gridspec(1, 2,  width_ratios=(5, 1),
                      left=0.1, right=0.9, top=0.95, bottom=0.15,
                      wspace=0.02)
# Create the Axes.
ax = fig.add_subplot(gs[0, 0])

ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)

# no labels
ax_histy.tick_params(axis="y", labelleft=False)

i = np.linspace(0,17,18)
# _sgimas = i ** (1/x_scale)

color_list = ['black','g','r','b','orange']

num_steps = 18
step_indices = torch.arange(num_steps)
rho = 7
sigmas = ((80) ** (1 / rho) + step_indices / (num_steps - 1) * ((0.002) ** (1 / rho) - (80) ** (1 / rho))) ** rho
sigmas_k = ((80) ** (1 / rho) + torch.arange(100000) / (100000 - 1) * ((0.002) ** (1 / rho) - (80) ** (1 / rho))) ** rho
k = 0
for sigma in sigmas_k:
    if sigma<=sigma_start and sigma>=sigma_end:
        k+=1
k = str(k/100000)[:4]
sigmas = sigmas ** (1/y_scale)
ax.plot(i, sigmas, label = r'EDM' + r', $r_p$' + f'= {k}', c = 'r')
ax.scatter(i, sigmas,s=5,c='r')

# ['gray','g','r','b','orange']
for (rho,color) in zip([1, 7],['b','orange']):
    sigmas = ((sigma_max) ** (1 / rho) + step_indices / (num_steps - 1) * ((sigma_min) ** (1 / rho) - (sigma_max) ** (1 / rho))) ** rho
    sigmas_k = ((sigma_max) ** (1 / rho) + torch.arange(100000) / (100000 - 1) * ((sigma_min) ** (1 / rho) - (sigma_max) ** (1 / rho))) ** rho
    k = 0
    for sigma in sigmas_k:
        if sigma<=sigma_start and sigma>=sigma_end:
            k+=1
    k = str(k/100000)[:4]
    sigmas = sigmas ** (1/y_scale)
    ax.plot(i, sigmas, label = r'PoDM ($\rho$' + f'={rho})' + r', $r_p$' + f'= {k}', c = color)
    ax.scatter(i, sigmas,s=5,c=color)


rho_3 = 3
sigmas_3 = ((sigma_max) ** (1 / rho_3) + step_indices / (num_steps - 1) * ((sigma_min) ** (1 / rho_3) - (sigma_max) ** (1 / rho_3))) ** rho_3
sigmas_k = ((sigma_max) ** (1 / rho_3) + torch.arange(100000) / (100000 - 1) * ((sigma_min) ** (1 / rho_3) - (sigma_max) ** (1 / rho_3))) ** rho_3

k_3 = 0
for sigma in sigmas_k:
    if sigma<=sigma_start and sigma>=sigma_end:
        k_3+=1
k_3 = str(k_3/100000)[:4]

rho_9 = 9
sigmas_9 = ((sigma_max) ** (1 / rho_9) + step_indices / (num_steps - 1) * ((sigma_min) ** (1 / rho_9) - (sigma_max) ** (1 / rho_9))) ** rho_9
sigmas_k = ((sigma_max) ** (1 / rho_9) + torch.arange(100000) / (100000 - 1) * ((sigma_min) ** (1 / rho_9) - (sigma_max) ** (1 / rho_9))) ** rho_9

k_9 = 0
for sigma in sigmas_k:
    if sigma<=sigma_start and sigma>=sigma_end:
        k_9+=1
k_9 = str(k_9/100000)[:4]

ax.fill_between(i, sigmas_3**(1/y_scale), sigmas_9**(1/y_scale), color='orange' , alpha=.2, linewidth=0, label = r'PoDM ($\rho\in[3,9])$'+ f', {k_3}' + r'$\leq r_p \leq $' + f'{k_9}' )


end_x = (sigma_end**(1 / rho) - sigma_max ** (1 /  rho))*(num_steps - 1)/( sigma_min ** (1 /  rho) -  sigma_max ** (1 /  rho))
start_x = (sigma_start**(1 / rho) - sigma_max ** (1 /  rho))*(num_steps - 1)/( sigma_min ** (1 /  rho) -  sigma_max ** (1 /  rho))

key_point_locs = [0.2, start_x, end_x, 16.3, 0 ]
key_points = [sigma_max+1, sigma_start+0.8, sigma_end+0.2, sigma_min+0.2, 2]
key_point_names = [r'$\sigma_{max}$', r'$\sigma_{start}$',r'$\sigma_{end}$',r'$\sigma_{min}$', 'Plausibility-relevant noises']

for loc, point, txt in zip(key_point_locs[:-1], key_points[:-1], key_point_names[:-1]):
    ax.annotate(txt, (loc, point ** (1/y_scale)))

ax.annotate(key_point_names[-1], (key_point_locs[-1], key_points[-1] ** (1/y_scale)))

ax.hlines(sigma_end**(1/y_scale),colors='gray', xmin=-1, xmax= 100, linestyles='dashed',alpha=0.5)
ax.hlines(sigma_start**(1/y_scale),colors='gray',xmin=-1, xmax= 100,linestyles='dashed',alpha=0.5)
# ax.fill_between(np.linspace(-0.5,19,19), np.ones(19)*sigma_end**(1/y_scale), np.ones(19)*sigma_start**(1/y_scale),color='r' , alpha=.2, linewidth=0)
# ax_histy.fill_between(np.linspace(0,2,100), np.ones(100)*sigma_end**(1/y_scale), np.ones(100)*sigma_start**(1/y_scale), alpha=.3, linewidth=0)

p_mean = math.log(sigma_end*sigma_start)/2
p_std = math.log(sigma_start/sigma_end)/2

# y = np.linspace(math.log(sigma_min), math.log(sigma_max), 100000)
# sigmas_training = stats.norm.pdf(y, p_mean, p_std)
 
# ax_histy.plot(sigmas_training, [math.exp(_) ** (1/y_scale) for _ in y], color= "#6495ED", label = 'SDM(ours)')

sigmas_training_hist= np.exp(np.random.randn(100000) * p_std + p_mean) ** (1/y_scale)
sigmas_training_hist_edm= np.exp(np.random.randn(100000) * 1.2 -1.2) ** (1/y_scale)
# ax_histy.boxplot(sigmas_training,showfliers=False)
num_bins = 1000
ax_histy.hist(sigmas_training_hist, num_bins, density=True, color='orange', alpha=0.5, orientation='horizontal',label='PoDM')
ax_histy.hist(sigmas_training_hist_edm, num_bins, density=True, color='r', alpha=0.5, orientation='horizontal',label='EDM')

x_ticks = np.array(['i=0'])
x_ticks = np.append(x_ticks, np.arange(1,18,1))
x_ticks = x_ticks.squeeze()
print(x_ticks)
ax.set_xlim(-0.5, 17.5)
ax.set_xticks(np.arange(0,18,1),x_ticks)

ax.set_ylim(ystick_labels[0]** (1/y_scale), ystick_labels[-1]** (1/y_scale))
ax.set_yticks([ystick_label ** (1/y_scale) for ystick_label in ystick_labels], ystick_labels)
ax.set_xlabel("sampling step i")
ax.set_ylabel(r'noise level $\sigma(t)$', loc='center',labelpad=-5)

ax_histy.set_ylim(ystick_labels[0]** (1/y_scale), ystick_labels[-1]** (1/y_scale))
ax_histy.set_yticks([ystick_label ** (1/y_scale) for ystick_label in ystick_labels], ystick_labels)
ax_histy.set_xlim(0,3)
ax_histy.set_xticks([])

# ax_histy.set_xlabel("training density",loc='right',labelpad=14)
# ax_histy.set_ylabel(r'distribution of noise scales in training', loc='center',labelpad=-80)
ax_histy.set_axis_off()

# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['right'].d
ax.legend(fontsize = 8)
ax_histy.legend(loc="upper left",fontsize = 8)
plt.show()