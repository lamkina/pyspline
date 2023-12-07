# External modules
import numpy as np

# First party modules
from pyspline.export import writeTecplot, writeTecplot1D
from pyspline.fitting import curveLMSApprox
from pyspline.projections import curveCurve, pointCurve

# Projection Tests
x = [0, 2, 3, 5]
y = [-2, 5, 3, 0]
z = [0, 0, 0, 0]

points = np.column_stack((x, y, z))
curve1, _ = curveLMSApprox(points=points, degree=3, nCtl=4, outputPrint=False)
writeTecplot(curve1, "curve1.dat")

x = [-2, 5, 2, 1]
y = [5, 1, 4, 2]
z = [3, 0, 1, 4]

points = np.column_stack((x, y, z))
curve2, _ = curveLMSApprox(points=points, degree=3, nCtl=4, outputPrint=False)
writeTecplot(curve2, "curve2.dat")

# Get the minimum distance distance between a point and each curve
x0 = np.array([4, 4, 3])
u1, d1 = pointCurve(x0, curve1, nIter=30, tol=1e-10, printLevel=0, u=0.5)
val1 = curve1(u1)  # Closest point on curve1

u2, d2 = pointCurve(x0, curve2, nIter=30, tol=1e-10, printLevel=0, u=1.0)
val2 = curve2(u2)  # Closest point on curve2

with open("projections.dat", "w") as file:
    data = np.row_stack((x0, val1))
    writeTecplot1D(file, "curve1_proj", data)

    data = np.row_stack((x0, val2))
    writeTecplot1D(file, "curve2_proj", data)

# Get the minimum distance between the two curves
s, t, D = curveCurve(curve1, curve2, nIter=30, tol=1e-10, printLevel=0)
val1 = curve1(s)
val2 = curve2(t)

with open("projections.dat", "a") as file:
    data = np.row_stack((val1, val2))
    writeTecplot1D(file, "curve1_curve2", data)
