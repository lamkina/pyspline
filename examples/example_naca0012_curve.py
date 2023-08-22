# External modules
import numpy as np

# First party modules
from pyspline.export import writeTecplot
from pyspline.fitting import curveLMSApprox

# Load naca0012 data
print("Naca 0012 data")
x, y = np.loadtxt("naca0012", unpack=True)

# Create the pointset for the airfoil
points = np.column_stack((x, y))

# Fit a BSpline to the airfoil data
airfoil = curveLMSApprox(points=points, degree=3, nCtl=11)

# Export the airfoil
writeTecplot(airfoil, "naca_data.dat")
