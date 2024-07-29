import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from scipy.stats import gamma
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

results_reshaped = np.load('drift_results_2.npy')
start_point = 0

sigmas = torch.linspace(0, 10, steps=100)
sigmas = sigmas+0.00001

sigmas = sigmas[start_point:]
results_reshaped = results_reshaped[start_point:]

sigma_start = 5.3
sigma_end = 0.6

sigma_max = sigma_start**2/sigma_end
sigma_min = sigma_end**2/sigma_start

y_scale = 1

num_steps = 10000
rho = 7
step_indices = torch.arange(num_steps)
end_point = 10.1

xstick_labels = [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 2, 10]
xstick_labels = [0.003, 0.01, 0.03, 0.1, 0.5, 2, 10]

sigmas_sampling = ((sigma_max) ** (1 / rho) + step_indices / (num_steps - 1) * ((sigma_min) ** (1 / rho) - (sigma_max) ** (1 / rho))) ** rho
sigmas_sampling = sigmas_sampling[sigmas_sampling<=end_point]
params = gamma.fit(sigmas_sampling)
dist = gamma(*params)
sigmas_sampling_y = dist.pdf(sigmas)


sigmas_sampling_EDM = ((80) ** (1 / rho) + step_indices / (num_steps - 1) * ((0.002) ** (1 / rho) - (80) ** (1 / rho))) ** rho
sigmas_sampling_EDM = sigmas_sampling_EDM[sigmas_sampling_EDM<=end_point]
params = gamma.fit(sigmas_sampling_EDM)
dist = gamma(*params)
sigmas_sampling_EDM_y = dist.pdf(sigmas)

p_mean = math.log(sigma_end*sigma_start)/2
p_std = math.log(sigma_start/sigma_end)/2
sigmas_training_hist= np.exp(np.random.randn(num_steps) * p_std + p_mean)
sigmas_training_hist = sigmas_training_hist[sigmas_training_hist<=end_point]

params = gamma.fit(sigmas_training_hist)
dist = gamma(*params)
sigmas_training_y = dist.pdf(sigmas)

p_mean = -1.2
p_std = 1.2
sigmas_training_EDM_hist= np.exp(np.random.randn(num_steps) * p_std + p_mean)
sigmas_training_EDM_hist = sigmas_training_EDM_hist[sigmas_training_EDM_hist<=10]

params = gamma.fit(sigmas_training_EDM_hist)
dist = gamma(*params)
sigmas_training_EDM_y = dist.pdf(sigmas)

x = sigmas.cpu()

y = []
dxdt = []
k = []
upper_y = results_reshaped.max()
for i, result in enumerate(results_reshaped):
    y.append((result).mean() ** (1/y_scale))
    dxdt.append((result).mean()* (-1) * x[i] ** (1/y_scale))
    k.append(1/x[i].numpy()**2)

delta_score = np.gradient(y, sigmas)

# Create a figure and axes
fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

# Plot the first function
# ax1.plot(x, delta_score, color='tab:orange', label = 'score function gradient')
ax1.plot(x, y, color='black',alpha=0.5, label = '2-Norm of score function')
# ax1.plot(x, dxdt, color='tab:red', label = 'x-gradient')
ax1.set_xlabel('noise level')
# ax1.set_ylabel('y1', color='b')
ax1.tick_params('y', colors='black')
ax1.set_xscale('log')
ax1.set_xticks(xstick_labels, xstick_labels)
ax1.set_ylabel('score')
ax1.set_yticks([])
ax1.set_xlim(0.0, 10)

# Create a second y-axis
ax2 = ax1.twinx()

density_ = True
bar_nr = 1000
# Plot the second function
ax2.hist(sigmas_sampling, bar_nr//10, density=density_, color='yellow', alpha=0.5, label='PoDM sampling density')
ax2.hist(sigmas_training_hist, bar_nr//5, density=density_, color='orange', alpha=0.3,label='PoDM training density')
ax2.tick_params('y', colors='blue')
ax2.set_yticks([])
# ax2.set_axis_off()
ax2.set_ylabel('density')

xstick_labels = [0.003, 0.01, 0.03, 0.1, 0.5, 2, 10]
# ax2.set_ylabel('y2', color='r')
ax3.plot(x, y, color='black', alpha=0.5, label = '2-Norm of score function')
ax3.set_xlabel('noise level')
ax3.tick_params('y', colors='black')
ax3.set_xscale('log')
ax3.set_xticks(xstick_labels, xstick_labels)
ax3.set_ylabel('score')
ax3.set_yticks([],[])
ax3.set_xlim(0.00, 10)

ax4 = ax3.twinx()

ax4.hist(sigmas_sampling_EDM, bar_nr, density=density_, color='red', alpha=0.5, label='EDM sampling density')
ax4.hist(sigmas_training_EDM_hist, bar_nr, density=density_, color='purple', alpha=0.5,label='EDM training density')

ax4.set_yticks([])
ax4.tick_params('y', colors='black')
ax4.set_ylabel('density')

# Get the handles and labels for all lines and histograms
lines, labels = ax1.get_legend_handles_labels()
hist, hist_labels = ax2.get_legend_handles_labels()

hist2, hist_labels2 = ax4.get_legend_handles_labels()

# Combine the handles and labels into one set
handles = lines + hist + hist2
labels1 = labels + hist_labels + hist_labels2

# handles2 = lines + hist2
# labels2 = labels + hist_labels2

# Add a legend
ax1.legend(handles, labels1, loc='upper right', fontsize = 8)
# ax1.set_title('PoDM')
# ax3.legend(handles2, labels2, loc='upper right')
# ax3.set_title('EDM')

y_scale = 1
plt.subplots_adjust(wspace=0.02, hspace = 0.5)
plt.show() 
# plt.savefig("alignment_score.pdf", format="pdf", bbox_inches="tight")

