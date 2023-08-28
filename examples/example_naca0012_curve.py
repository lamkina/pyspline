# External modules
import numpy as np

# First party modules
from pyspline.bspline import BSplineCurve
from pyspline.export import writeIGES, writeTecplot
from pyspline.fitting import curveLMSApprox

# Load naca0012 data
print("Naca 0012 data")
x, y = np.loadtxt("naca0012", unpack=True)

# Create the pointset for the airfoil
points = np.column_stack((x, y))

# Fit a BSpline to the airfoil data
airfoil = curveLMSApprox(points=points, degree=3, nCtl=50, maxIter=1, tol=1e-6, nParamIters=1)
airfoil2 = curveLMSApprox(points=points, degree=3, nCtl=20, maxIter=1, tol=1e-6, nParamIters=1)
airfoil2.ctrlPnts = airfoil2.ctrlPnts * 1.5

# Export the airfoil
# writeTecplot(airfoil, "naca_data.dat")
writeIGES("naca0012.iges", [airfoil, airfoil2])
