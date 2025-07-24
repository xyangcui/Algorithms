import numpy as np
import matplotlib.pyplot as plt
from fun_dynsys_bivariate_analysis import fun_dynsys_bivariate_analysis

# INPUTS
quanti = 0.98

# Generate random 'x' and 'y' tensors
x_lonlat = np.random.randn(1000, 10, 15)
y_lonlat = np.random.randn(1000, 10, 15)

# Define time, lon, and lat
time = np.arange(1, 1001)
lon = np.arange(1, 11)
lat = np.arange(1, 16)

# Reshape tensors to obtain time * space matrices
x = x_lonlat.reshape(len(time), len(lon) * len(lat))
y = y_lonlat.reshape(len(time), len(lon) * len(lat))

# Compute dynamical quantities
D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha = fun_dynsys_bivariate_analysis(x, y, quanti)

# EXAMPLE PLOTS
# Plot the different monovariate and bivariate dynamical quantities
plt.figure()

plt.subplot(3, 1, 1)
plt.plot(time, D1_x)
plt.plot(time, D1_y)
plt.plot(time, D1_con)
plt.title('a) Local dimensions')
plt.xlabel('time')
plt.ylabel('D1')
plt.legend(['x', 'y', 'co-dimension'])

plt.subplot(3, 1, 2)
plt.plot(time, theta_x)
plt.plot(time, theta_y)
plt.plot(time, theta_con)
plt.title('b) Local inverse persistence')
plt.xlabel('time')
plt.ylabel('theta')
plt.legend(['x', 'y', 'co-persistence'])

plt.subplot(3, 1, 3)
plt.plot(time, alpha)
plt.title('c) Co-recurrence coefficient')
plt.xlabel('time')
plt.ylabel('alpha')

plt.show()
