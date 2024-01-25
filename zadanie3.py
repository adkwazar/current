import numpy as np
import matplotlib.pyplot as plt

x = np.array([]) #wpisz do [], po przecinku kolejne wartosci
y = np.array([]) #wpisz do [], po przecinku kolejne wartosci


print(np.corrcoef(x,y)[0][1])

plt.scatter(x,y)
plt.show()