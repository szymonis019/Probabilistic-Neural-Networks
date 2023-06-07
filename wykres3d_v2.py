import numpy as np
from sklearn.model_selection import StratifiedKFold
from neupy.algorithms import PNN
import hickle as hkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('haberman.hkl')
y_t -= 1
x = x_norm.T
y_t = np.squeeze(y_t)
data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

PK_vec = []

# Tablica zmiennych std
std_values = []
std = 3.0
while std >= 0.001:
    std_values.append(std)
    std -= 0.1  # Krok dla pętli

verbose_values = [True, False]  # Przykładowe wartości verbose
batch_size_values = np.arange(128, 0, -1)  # Przykładowe wartości batch_size

std_mesh, verbose_mesh, batch_size_mesh = np.meshgrid(std_values, verbose_values, batch_size_values)

PK_mesh = np.zeros_like(std_mesh)

for i in range(std_mesh.shape[0]):
    for j in range(std_mesh.shape[1]):
        for k in range(std_mesh.shape[2]):
            std = std_mesh[i, j, k]
            verbose = bool(verbose_mesh[i, j, k])
            batch_size = batch_size_mesh[i, j, k]

            PK_values = []
            for l, (train, test) in enumerate(skfold.split(data, target), start=0):
                x_train, x_test = data[train], data[test]
                y_train, y_test = target[train], target[test]

                pnn_network = PNN(std=std, verbose=verbose, batch_size=batch_size)
                pnn_network.train(x_train, y_train)
                result = pnn_network.predict(x_test)
                n_test_samples = test.size
                PK = np.sum(result == y_test) / n_test_samples
                PK_values.append(PK)

            PK_mean = np.mean(PK_values)
            PK_mesh[i, j, k] = PK_mean

fig = plt.figure(figsize=(8, 8))
fig2 = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')

batch_size_mesh_2d = batch_size_mesh.reshape(-1)
std_mesh_2d = std_mesh.reshape(-1)
verbose_mesh_2d = verbose_mesh.reshape(-1)
PK_mesh_2d = PK_mesh.reshape(-1)

false_index=verbose_mesh_2d.tolist().index(False)
#print(false_index)
ax.scatter(batch_size_mesh_2d[:false_index], std_mesh_2d[:false_index], verbose_mesh_2d[:false_index], c=PK_mesh_2d[:false_index], cmap='viridis')
ax2.scatter(batch_size_mesh_2d[false_index:], std_mesh_2d[false_index:], verbose_mesh_2d[false_index:], c=PK_mesh_2d[false_index:], cmap='viridis')

ax.set_xlabel('batch_size')
ax.set_zlabel('verbose')
ax.set_ylabel('std')
ax.set_title('Zależność PK od batch_size, verbose i std')

ax2.set_xlabel('batch_size')
ax2.set_zlabel('verbose')
ax2.set_ylabel('std')
ax2.set_title('Zależność PK od batch_size, verbose i std')

#print(batch_size_mesh_2d)
#print(std_mesh_2d)
#print(verbose_mesh_2d)