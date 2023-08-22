# External modules
import numpy as np

# First party modules
from pyspline.export import writeTecplot
from pyspline.fitting import curveLMSApprox
from pyspline.projections import curveCurve, pointCurve

# Projection Tests
print("Projection Tests")
x = [0, 2, 3, 5]
y = [-2, 5, 3, 0]
z = [0, 0, 0, 0]

points = np.column_stack((x, y, z))
curve1 = curveLMSApprox(points=points, degree=3, nCtl=4)
writeTecplot(curve1, "curve1.dat")

x = [-2, 5, 2, 1]
y = [5, 1, 4, 2]
z = [3, 0, 1, 4]

points = np.column_stack((x, y, z))
curve2 = curveLMSApprox(points=points, degree=3, nCtl=4)
writeTecplot(curve2, "curve2.dat")

# Get the minimum distance distance between a point and each curve
x0 = np.array([4, 4, 3])
u1, d1 = pointCurve(x0, curve1, nIter=10, tol=1e-10, u=0.5)
val1 = curve1(u1)  # Closest point on curve1

u2, d2 = pointCurve(x0, curve2, nIter=10, tol=1e-10, u=1.0)
val2 = curve2(u2)  # Closest point on curve2

# Output the data
f = open("projections.dat", "w")
f.write('VARIABLES = "X", "Y","Z"\n')
f.write("Zone T=curve1_proj I=2 \n")
f.write("DATAPACKING=POINT\n")
f.write("%f %f %f\n" % (x0[0], x0[1], x0[2]))
f.write("%f %f %f\n" % (val1[0], val1[1], val1[2]))

f.write("Zone T=curve2_proj I=2 \n")
f.write("DATAPACKING=POINT\n")
f.write("%f %f %f\n" % (x0[0], x0[1], x0[2]))
f.write("%f %f %f\n" % (val2[0], val2[1], val2[2]))

# Get the minimum distance between the two curves
s, t, D = curve1.projectCurve(curve2)
val1 = curve1(s)
val2 = curve2(t)

f.write("Zone T=curve1_curve2 I=2 \n")
f.write("DATAPACKING=POINT\n")
f.write("%f %f %f\n" % (val1[0], val1[1], val1[2]))
f.write("%f %f %f\n" % (val2[0], val2[1], val2[2]))
