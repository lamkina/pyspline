# External modules
import numpy as np

# First party modules
from pyspline.bspline import BSplineSurface
from pyspline.fitting import computeKnotVecLMS, curveLMSApprox
from pyspline.pointInversion import pointInversionSurface

# Create a NACA0012 airfoil as the curve
x, y = np.loadtxt("naca0012", unpack=True)
points = np.column_stack((x, y, np.zeros(len(x))))
airfoil = curveLMSApprox(points=points, degree=3, nCtl=50, maxIter=1, tol=1e-6, nParamIters=1)

# We can create a simple wing-like surface by translating the airfoil control points twice in the z-direction
ctrlPnts = np.zeros((airfoil.nCtl, 3, 3))
ctrlPnts[:, 0] = airfoil.ctrlPnts  # First cross section
ctrlPnts[:, 1] = airfoil.ctrlPnts + np.array([0, 0, 1])  # Second cross section
ctrlPnts[:, 2] = airfoil.ctrlPnts + np.array([0, 0, 2])  # Third cross section

uKnotVec = airfoil.knotVec  # u-knot vector comes from the airfoil
v = np.linspace(0, 1, 3)
vDegree = 1
vKnotVec = computeKnotVecLMS(v, 3, 3, vDegree)  # Linear knot vector in the v-direction
wing = BSplineSurface(airfoil.degree, vDegree, ctrlPnts, uKnotVec, vKnotVec)

# Lets pick some x and z points between 0 and 1
x1Coords = np.linspace(0.01, 0.99, 100)
x2Coords = np.linspace(0.01, 0.99, 100)

# Solve for the parameteric u and v coordinates
uStar, vStar = pointInversionSurface(
    x1Coords,
    x2Coords,
    ("x", "z"),
    wing,
    0,  # Lower bounds for the u-coordinate
    1,  # Upper bounds for the u-coordinate
    0,  # Lower bounds for the v-coordinate
    1,  # Upper bounds for the v-coordinate
    nIter=25,
    tol=1e-6,
    printLevel=1,
    u0=x1Coords.copy(),  # Copying the coordinates is a good enough initial guess (typically not true)
    v0=x2Coords.copy(),  # Copying the coordinates is a good enough initial guess (typically not true)
)

# Print out the u, v and (x,y,z) values
for u, v in zip(uStar, vStar):
    print(f"U: {u:.10f}, V: {v:.10f}, Point: {wing(u, v)}")
