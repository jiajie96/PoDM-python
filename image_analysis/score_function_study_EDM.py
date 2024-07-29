import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from scipy.stats import gamma

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
end_point = 80
end_point_2 = 80

xstick_labels = [0.001, 0.005, 0.02, 0.05, 0.1, 0.5, 2, 10]
# [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 2, 10]

sigmas_sampling = ((sigma_max) ** (1 / rho) + step_indices / (num_steps - 1) * ((sigma_min) ** (1 / rho) - (sigma_max) ** (1 / rho))) ** rho
sigmas_sampling = sigmas_sampling[sigmas_sampling<=9]
params = gamma.fit(sigmas_sampling)
dist = gamma(*params)
sigmas_sampling_y = dist.pdf(sigmas)


sigmas_sampling_EDM = ((80) ** (1 / rho) + step_indices / (num_steps - 1) * ((0.002) ** (1 / rho) - (80) ** (1 / rho))) ** rho
# sigmas_sampling_EDM = sigmas_sampling_EDM[sigmas_sampling_EDM<=end_point]
params = gamma.fit(sigmas_sampling_EDM)
dist = gamma(*params)
sigmas_sampling_EDM_y = dist.pdf(sigmas)

p_mean = math.log(sigma_end*sigma_start)/2
p_std = math.log(sigma_start/sigma_end)/2
sigmas_training_hist= np.exp(np.random.randn(num_steps) * p_std + p_mean)
sigmas_training_hist = sigmas_training_hist[sigmas_training_hist<=10]

params = gamma.fit(sigmas_training_hist)
dist = gamma(*params)
sigmas_training_y = dist.pdf(sigmas)

p_mean = -1.2
p_std = 1.2
sigmas_training_EDM_hist= np.exp(np.random.randn(num_steps) * p_std + p_mean)
sigmas_training_EDM_hist = sigmas_training_EDM_hist[sigmas_training_EDM_hist<=end_point]

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
fig, ax1 = plt.subplots(figsize=(6, 4))

# Plot the first function
# ax1.plot(x, delta_score, color='tab:orange', label = 'score function gradient')
ax1.plot(x, y, color='tab:blue', label = 'Score function')
# ax1.plot(x, dxdt, color='tab:red', label = 'x-gradient')
ax1.set_xlabel('noise level')
# ax1.set_ylabel('y1', color='b')
ax1.tick_params('y', colors='b')
ax1.set_xscale('log')
ax1.set_xticks(xstick_labels, xstick_labels)
ax1.set_ylabel('score')
ax1.set_xlim(0.001, 10)

# Create a second y-axis
ax2 = ax1.twinx()

density_ = True
bar_nr = 1000
# Plot the second function
# ax2.hist(sigmas_sampling, bar_nr*2, density=density_, color='orange', alpha=0.3, label='PoDM sampling density')
# ax2.hist(sigmas_training_hist, bar_nr//5, density=density_, color='red', alpha=0.3,label='PoDM training density')

ax2.hist(sigmas_sampling_EDM, bar_nr, density=density_, color='green', alpha=0.3, label='EDM sampling density')
ax2.hist(sigmas_training_EDM_hist, bar_nr, density=density_, color='blue', alpha=0.3,label='EDM training density')

# ax2.plot(x, sigmas_training_y, color='black', label = 'PoDM training density')
# ax2.plot(x, sigmas_training_EDM_y, color='yellow', label = 'EDM training density')
# ax2.plot(x, sigmas_sampling_y, color='tab:red', label = 'PoDM sampling density')
# ax2.plot(x, sigmas_sampling_EDM_y, color='tab:blue', label = 'EDM sampling density')

# ax2.set_ylabel('y2', color='r')
ax2.tick_params('y', colors='r')
ax2.set_ylabel('density')

# Get the handles and labels for all lines and histograms
lines, labels = ax1.get_legend_handles_labels()
hist, hist_labels = ax2.get_legend_handles_labels()

# Combine the handles and labels into one set
handles = lines + hist
labels = labels + hist_labels

# Add a legend
ax1.legend(handles, labels, loc='upper right')

# plt.plot(x, y, label='std line')

ystick_labels = [0, 0.2,0.4,0.6,0.8,1]
xstick_labels = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.5, 2, 10]
# plt.ylim(ystick_labels[0]** (1/y_scale), ystick_labels[-1]** (1/y_scale))
# plt.ylim(0, 5)
# plt.xlim(0.002, 10)
y_scale = 1
# plt.yticks([ystick_label ** (1/y_scale) for ystick_label in ystick_labels], ystick_labels)
# plt.xticks(np.log(xstick_labels), xstick_labels)
# plt.xscale('log')
# plt.legend()
# plt.title('Alignment Test')
# plt.title('Score function  with logarithmic x-axis')
# plt.xlabel('noise level')
# plt.xscale('log')
# plt.show()
plt.savefig("alignment_EDM_score.pdf", format="pdf", bbox_inches="tight")



# y =[result.std() for result in results_reshaped]

# sum_ = [k.cpu()/20 for k in results[0]]
# upper = results_reshaped.max(axis=1)
# lower = results_reshaped.min(axis=1)

# plt.figure(figsize=(8, 3)) 
# plt.hist(sigmas_training_hist, 1000, density=True, color='red', alpha=0.3,label='PoDM training density')
# plt.hist(sigmas_sampling, 1000, density=True, color='orange', alpha=0.5, label='PoDM sampling density')
# plt.hist(sigmas_sampling_EDM, 1000, density=True, color='blue', alpha=0.5, label='EDM sampling density')
# plt.plot(x, delta_score, color='tab:orange', label = 'score function gradient')

# plt.plot(x, y1, color='tab:red', label = 'y1')
# plt.plot(x, sigmas_training_y, color='black', label = 'PoDM training density')
# plt.plot(x, sigmas_training_EDM_y, color='yellow', label = 'EDM training density')
# plt.plot(x, sigmas_sampling_y, color='tab:red', label = 'PoDM sampling density')
# plt.plot(x, sigmas_sampling_EDM_y, color='tab:blue', label = 'EDM sampling density')
# plt.plot(x[1:], k[1:], color='r')
# plt.plot(x[start_point:], upper[start_point:], color='tab:blue', alpha=0.5)
# plt.plot(x[start_point:], lower[start_point:], color='tab:blue', alpha=0.5)
# plt.fill_between(x[start_point:], lower[start_point:], upper[start_point:], alpha=0.5)

