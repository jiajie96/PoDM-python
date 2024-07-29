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

x = np.linspace(np.log(0.02), np.log(100), 10000)
y = np.exp(x)

# y = 1/x

plt.hist(y,1000,density = True,)

x = np.random.random(10000) * 1.2 -1.2
y = np.exp(x)

plt.hist(y,1000,density = True,)

# x = np.linspace(0.00001, 1, 10000)
# y = 1/x

# plt.hist(y,1000,density = True,)

plt.show()
