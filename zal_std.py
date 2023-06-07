import numpy as np
from sklearn.model_selection import StratifiedKFold
from neupy.algorithms import PNN
import hickle as hkl
import matplotlib.pyplot as plt

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('haberman.hkl')
y_t -= 1
x = x_norm.T
y_t = np.squeeze(y_t)
data = x
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = []

#tablica zmiennych std
std_values = []
std = 3.0
while std >= 0.001:
    std_values.append(std)
    std -= 0.1 #krok dla pęli

for std in std_values:
    PK_values = []
    for i, (train, test) in enumerate(skfold.split(data, target), start=0):
        x_train, x_test = data[train], data[test]
        y_train, y_test = target[train], target[test]

        pnn_network = PNN(std=std, verbose=True)
        pnn_network.train(x_train, y_train)
        result = pnn_network.predict(x_test)
        n_test_samples = test.size
        PK = np.sum(result == y_test) / n_test_samples
        PK_values.append(PK)

    PK_mean = np.mean(PK_values)
    PK_vec.append(PK_mean)

# Wykres
plt.plot(std_values, PK_vec)
plt.xlabel('std')
plt.ylabel('PK')
plt.title('Zależność PK od wartości std')
plt.show()

max_PK_index = np.argmax(PK_vec)
max_PK_std = std_values[max_PK_index]
max_PK_value = PK_vec[max_PK_index]

print("Maksymalna wartość PK:")
print("std = {:.10f}".format(max_PK_std))
print("PK = {:.10f}".format(max_PK_value))

print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))
PK = np.mean(PK_vec)
print("PK {}".format(PK))