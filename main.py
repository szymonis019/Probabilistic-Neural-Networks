import numpy as np
from sklearn.model_selection import StratifiedKFold
from neupy.algorithms import PNN
import hickle as hkl
import matplotlib.pyplot as plt

std_values = [0.1, 0.5, 1.0, 1.5]  # Przykładowe wartości std
verbose_values = [False, True]  # Przykładowe wartości verbose
batch_size_values = [16, 32, 64, 128]  # Przykładowe wartości batch_size

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for std in std_values:
    for verbose in verbose_values:
        for batch_size in batch_size_values:
            x, y_t, x_norm, x_n_s, y_t_s = hkl.load('haberman.hkl')
            y_t -= 1
            x = x_norm.T
            y_t = np.squeeze(y_t)
            data = x
            target = y_t
            CVN = 10
            skfold = StratifiedKFold(n_splits=CVN)
            PK_vec = np.zeros(CVN)

            for i, (train, test) in enumerate(skfold.split(data, target), start=0):
                x_train, x_test = data[train], data[test]
                y_train, y_test = target[train], target[test]
                pnn_network = PNN(std=std, verbose=verbose, batch_size=batch_size)
                pnn_network.train(x_train, y_train)
                result = pnn_network.predict(x_test)
                n_test_samples = test.size
                PK_vec[i] = np.sum(result == y_test) / n_test_samples

            ax.scatter(std, verbose, batch_size, c=np.mean(PK_vec), cmap='viridis')

ax.set_xlabel('std')
ax.set_ylabel('verbose')
ax.set_zlabel('batch_size')
ax.set_title('Zależność PK od std, verbose i batch_size')
plt.show()