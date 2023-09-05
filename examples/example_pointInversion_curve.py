# External modules
import numpy as np

# First party modules
from pyspline.fitting import curveLMSApprox
from pyspline.pointInversion import pointInversionCurve

# Create a NACA0012 airfoil as the curve
x, y = np.loadtxt("naca0012", unpack=True)
points = np.column_stack((x, y))
airfoil = curveLMSApprox(points=points, degree=3, nCtl=50, maxIter=1, tol=1e-6, nParamIters=1)

# Lets pick an x-point between 0 and 1
xStar = np.linspace(0.01, 0.99, 10)

# Now lets find the parameteric u-coordinate that corresponds to this x-value
uStar = pointInversionCurve(xStar, "x", airfoil, 0, 1, nIter=100, tol=1e-6, printLevel=1, u0=xStar.copy())
print(f"U: {uStar}, X: {airfoil(uStar)}")
