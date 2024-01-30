import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras


x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()  