import numpy as np
import torch
import matplotlib.pyplot as plt

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

num_steps = 18
step_indices = torch.arange(num_steps)
sigma_max = 80
sigma_min = 0.002


sigmas_trainings =[]

p_std = 1.2
p_mean = 0.5
sigmas_training = (torch.randn(1000) * p_std + p_mean).exp()
sigmas_trainings.append(sigmas_training ** (1/2))
p_mean = -1.2
sigmas_training = (torch.randn(1000) * p_std + p_mean).exp()
sigmas_trainings.append(sigmas_training ** (1/2))

plt.figure(figsize=(12, 12))

for i, rho_index in enumerate([7, 5, 3]):
    sigmas = (sigma_max ** (1 / rho_index) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho_index) - sigma_max ** (1 / rho_index))) ** rho_index
    # sigmas = sigmas ** (1/2)
    (name, linestyle)  = linestyle_tuple[i]
    if rho_index == 7:
        label = r"EDM: Sampling with $\rho\ =\ $" + str(rho_index)
        print(sigmas)
    elif rho_index == 5:
        label = r"SDM: Sampling with $\rho\ =\ $" + str(rho_index)
        print(sigmas)
    else:

        label = r"Sampling with $\rho\ =\ $" + str(rho_index)
    plt.plot(step_indices+1, sigmas, label = label,  linestyle=linestyle)

plt.xlim(1, 18)
plt.xticks(np.arange(1,19,1))

ystick_labels = [0,0.2,1,5,10,20,40,80]
plt.yticks([ystick_label ** (1/2) for ystick_label in ystick_labels], ystick_labels)


rho_index = 5
sigma_semantic_max = 5
co_step_indice = (sigma_semantic_max**(1 / rho_index) - sigma_max ** (1 /  rho_index))*(num_steps - 1)/( sigma_min ** (1 /  rho_index) -  sigma_max ** (1 /  rho_index))

step_indices_interpolation = np.arange(num_steps)/num_steps*(num_steps - co_step_indice) + co_step_indice
sigmas_interpolation = (sigma_max ** (1 / rho_index) + step_indices_interpolation / (num_steps - 1) * (sigma_min ** (1 / rho_index) - sigma_max ** (1 / rho_index))) ** rho_index
sigmas_interpolation = sigmas_interpolation ** (1/2)

plt.scatter(step_indices_interpolation+1, sigmas_interpolation, label = 'noise schedule for semantic reconstruction')

plt.grid(True)
plt.legend()
plt.xlabel('Timesteps', loc='right')
plt.ylabel(r'Noise scale $\sigma$',loc='top')


plt.twiny()
plt.boxplot(sigmas_trainings, positions=[11, 13 ],labels=[' ', ' '], widths= [1, 1], showfliers=False)
plt.text(11.2, 3.65
         , r'SDM Training')
plt.text(13.2, 1.6, r'EDM Training')
plt.xlim(1, 18)
plt.yticks([ystick_label ** (1/2) for ystick_label in ystick_labels], ystick_labels)

plt.savefig('sigmas_p_mean.svg', bbox_inches='tight')
plt.show()