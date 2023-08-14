import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# data load
H_cond_map = np.load('res_H_state/H_cond_map.npy')
H_max_map = np.load('res_H_state/H_max_map.npy')
M_max_map = np.load('res_H_state/M_max_map.npy')

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.show()