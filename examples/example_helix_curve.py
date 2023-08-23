# External modules
import numpy as np

# First party modules
from pyspline.export import writeTecplot
from pyspline.fitting import curveLMSApprox

# Get some Helix-like data
n = 100
theta = np.linspace(0.0000, 2 * np.pi, n)
x = np.cos(theta)
y = np.sin(theta)
z = np.linspace(0, 1, n)
print("Helix Data")
points = np.column_stack((x, y, z))
curve = curveLMSApprox(points=points, degree=3, nCtl=16)
writeTecplot(curve, "helix.dat")
