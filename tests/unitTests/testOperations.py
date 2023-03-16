# Standard Python modules
from copy import deepcopy
import unittest

# External modules
import numpy as np
from numpy.testing import assert_allclose

# First party modules
from pyspline import BSplineCurve, BSplineSurface, NURBSCurve, NURBSSurface, operations
from pyspline.export import writeTecplot


class TestOperations(unittest.TestCase):
    N_PROCS = 1

    @classmethod
    def setUpClass(self) -> None:
        # Create a bspline curve
        knotVec = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
        degree = 3
        ctrlPnts = np.array([[0.0, 0.0], [1.0, 1.0], [1.5, 1.5], [2.0, 4.0], [3.0, 4.0], [3.5, 3.0], [4.0, 1.0]])
        self.bSplineCurve = BSplineCurve(degree, knotVec, ctrlPnts)

        # Create a nurbs curve
        knotVec = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])
        degree = 2
        ctrlPntsW = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [0.0, 1.0, 0.0, 1.0],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [-1.0, 0.0, 0.0, 1.0],
                [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [0.0, -1.0, 0.0, 1.0],
                [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        self.nurbsCurve = NURBSCurve(degree, knotVec, ctrlPntsW)

        # Create a bspline surface
        uKnotVec = np.array([0, 0, 0, 1 / 2, 1, 1, 1])
        vKnotVec = np.array([0, 0, 0, 1, 1, 1])
        uDegree = 2
        vDegree = 2
        ctrlPnts = np.zeros((4, 3, 3))
        ctrlPnts[0, 0] = np.array([0, 0, 0])
        ctrlPnts[0, 1] = np.array([0, 2, 2])
        ctrlPnts[0, 2] = np.array([0, 4, 0])
        ctrlPnts[1, 0] = np.array([3, 0, 3])
        ctrlPnts[1, 1] = np.array([3, 2, 5])
        ctrlPnts[1, 2] = np.array([3, 4, 3])
        ctrlPnts[2, 0] = np.array([6, 0, 3])
        ctrlPnts[2, 1] = np.array([6, 2, 5])
        ctrlPnts[2, 2] = np.array([6, 4, 3])
        ctrlPnts[3, 0] = np.array([9, 0, 0])
        ctrlPnts[3, 1] = np.array([9, 2, 2])
        ctrlPnts[3, 2] = np.array([9, 4, 0])

        self.bSplineSurf = BSplineSurface(uDegree, vDegree, ctrlPnts, uKnotVec, vKnotVec)

        # Create a nurbs surface
        uKnotVec = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])
        vKnotVec = np.array([0, 0, 1 / 2, 1, 1])
        uDegree = 2
        vDegree = 1
        ctrlPntsW = np.zeros((9, 3, 4))
        ctrlPntsW[:, 0] = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [0.0, 1.0, 0.0, 1.0],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [-1.0, 0.0, 0.0, 1.0],
                [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [0.0, -1.0, 0.0, 1.0],
                [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )

        ctrlPntsW[:, 1] = np.array(
            [
                [1.0, 0.0, 2.0, 1.0],
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 2.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [0.0, 1.0, 2.0, 1.0],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 2.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [-1.0, 0.0, 2.0, 1.0],
                [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 2.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [0.0, -1.0, 2.0, 1.0],
                [np.sqrt(2) / 2, -np.sqrt(2) / 2, 2.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [1.0, 0.0, 2.0, 1.0],
            ]
        )

        ctrlPntsW[:, 2] = np.array(
            [
                [1.0, 0.0, 4.0, 1.0],
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 4.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [0.0, 1.0, 4.0, 1.0],
                [-np.sqrt(2) / 2, np.sqrt(2) / 2, 4.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [-1.0, 0.0, 4.0, 1.0],
                [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 4.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [0.0, -1.0, 4.0, 1.0],
                [np.sqrt(2) / 2, -np.sqrt(2) / 2, 4.0 * np.sqrt(2) / 2, np.sqrt(2) / 2],
                [1.0, 0.0, 4.0, 1.0],
            ]
        )

        self.nurbsSurf = NURBSSurface(uDegree, vDegree, ctrlPntsW, uKnotVec, vKnotVec)

    def testDecomposeCurve(self):
        newCurves = operations.decomposeCurve(self.bSplineCurve)
        self.assertEqual(len(newCurves), 4)
        for curve in newCurves:
            self.assertEqual(curve.degree, 3)
            self.assertEqual(len(curve.knotVec), 8)
            self.assertTrue(isinstance(curve, BSplineCurve))
            self.assertFalse(curve.rational)

        newCurves = operations.decomposeCurve(self.nurbsCurve)
        self.assertEqual(len(newCurves), 4)
        for curve in newCurves:
            self.assertEqual(curve.degree, 2)
            self.assertEqual(len(curve.knotVec), 6)
            self.assertTrue(isinstance(curve, NURBSCurve))
            self.assertTrue(curve.rational)

    def testCombineCurves(self):
        # Create 4 decomposed bezier line segments from a BSpline
        curveList = []
        for i in range(4):
            curve = BSplineCurve(1, np.array([0, 0, 1, 1]), np.array([[i, i], [i + 1, i + 1]]))
            curveList.append(curve)

        knotVec, ctrlPnts, weights, knots = operations.combineCurves(curveList)
        assert_allclose(knotVec, np.array([0, 0, 1, 2, 3, 4, 4]))  # Test the knot vector generation (non-normalized)
        assert_allclose(ctrlPnts, np.array([[i, i] for i in range(4 + 1)]))  # Test the control point generation
        assert_allclose(weights, np.ones(len(ctrlPnts)))  # Test the weights get the correct default values
        assert_allclose(knots, np.array([1, 2, 3]))  # Test the internal knot generation

        # Create 4 decomposed bezier line segments from a nurbs circle
        curveList = []
        for i in range(0, 7, 2):
            ctrlPntsW = self.nurbsCurve.ctrlPntsW[i : i + 3]
            curve = NURBSCurve(2, np.array([0, 0, 0, 1, 1, 1]), ctrlPntsW)
            curveList.append(curve)

        knotVec, ctrlPnts, weights, knots = operations.combineCurves(curveList)
        assert_allclose(knotVec, np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]))
        assert_allclose(ctrlPnts, self.nurbsCurve.ctrlPnts)
        assert_allclose(weights, self.nurbsCurve.weights)
        assert_allclose(knots, np.array([1, 2, 3]))

    def testElevateDegreeCurve(self):
        curve = deepcopy(self.bSplineCurve)  # copy curve so we dont change the original

        # Elevate degree by 1
        operations.elevateDegree(curve, [1])
        self.assertEqual(curve.degree, 4)

        # Test multi-degree elevation
        operations.elevateDegree(curve, [2])
        self.assertEqual(curve.degree, 6)

        curve = deepcopy(self.nurbsCurve)  # copy curve so we dont change the original

        # Elevate degree by 1
        operations.elevateDegree(curve, [1])
        self.assertEqual(curve.degree, 3)

        writeTecplot(curve, "/home/mdolabuser/shared/repos/pyloft/pyloft/OUTPUT/nurbs_elev_curve.dat")
        print("Degree 3")
        print(np.linalg.norm(curve.data - np.array([0.0, 0.0, 0.0]), axis=1))

        # Test multi-degree elevation
        operations.elevateDegree(curve, [2])
        self.assertEqual(curve.degree, 5)

    def testElevateDegreeSurface(self):
        pass

    def testElevateDegreeVolume(self):
        pass

    # def testReduceDegreeCurve(self):
    #     # Test degree reduction for BSplines
    #     knotVec = np.array(
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #     )
    #     ctrlPnts = np.array(
    #         [
    #             [0.0, 0.0],
    #             [0.5, 0.5],
    #             [0.775, 0.775],
    #             [0.9395833333333334, 0.9729166666666667],
    #             [1.0695601851851853, 1.1839120370370373],
    #             [1.7347222222222223, 2.5819444444444444],
    #             [1.866666666666667, 2.847685185185185],
    #             [2.576388888888889, 3.5875],
    #             [2.7391203703703697, 3.575],
    #             [3.4833333333333334, 2.8625],
    #             [3.6, 2.5500000000000003],
    #             [3.75, 1.9999999999999998],
    #             [4.0, 1.0],
    #         ]
    #     )
    #     curve = BSplineCurve(6, knotVec, ctrlPnts)
    #     operations.reduceDegree(curve, [1])
    #     self.assertEqual(curve.degree, 5)

    #     # Test degree reduction for NURBS
    #     knotVec = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0])
    #     ctrlPntsW = np.array(
    #         [
    #             [1.0, 0.0, 0.0, 1.0],
    #             [0.804737854124365, 0.47140452079103173, 0.0, 0.804737854124365],
    #             [0.47140452079103173, 0.804737854124365, 0.0, 0.804737854124365],
    #             [-0.47140452079103173, 0.804737854124365, 0.0, 0.804737854124365],
    #             [-0.804737854124365, 0.47140452079103173, 0.0, 0.804737854124365],
    #             [-0.804737854124365, -0.47140452079103173, 0.0, 0.804737854124365],
    #             [-0.47140452079103173, -0.804737854124365, 0.0, 0.804737854124365],
    #             [0.47140452079103173, -0.804737854124365, 0.0, 0.804737854124365],
    #             [0.804737854124365, -0.47140452079103173, 0.0, 0.804737854124365],
    #             [1.0, 0.0, 0.0, 1.0],
    #         ]
    #     )
    #     curve = NURBSCurve(3, knotVec, ctrlPntsW)
    #     curve.computeData()
    #     print(np.linalg.norm(curve.data - np.array([0.0, 0.0, 0.0]), axis=1))
    #     operations.reduceDegree(curve, [1])
    #     self.assertEqual(curve.degree, 2)
    #     # writeTecplot(curve, "/home/mdolabuser/shared/pyLoft/pyloft/OUTPUT/testnurbscurve.dat")
    #     print(curve.knotVec)
    #     print(curve.ctrlPntsW)
    #     print(curve.ctrlPnts)
    #     print(np.linalg.norm(curve.data - np.array([0.0, 0.0, 0.0]), axis=1))

    #     self.nurbsCurve.computeData()
    #     print(np.linalg.norm(self.nurbsCurve.data - np.array([0.0, 0.0, 0.0]), axis=1))
    #     # writeTecplot(self.nurbsCurve, "/home/mdolabuser/shared/pyLoft/pyloft/OUTPUT/testnurbscircle.dat")

    # def testReduceDegreeSurface(self):
    #     pass

    # def testReduceDegreeVolume(self):
    #     pass

    # def testRefineKnotVecCurve(self):
    #     pass

    # def testRefineKnotVecSurface(self):
    #     pass

    # def testRefineKnotVecVolume(self):
    #     pass

    # def testInsertKnotCurve(self):
    #     pass

    # def testInsertKnotSurface(self):
    #     pass

    # def testInsertKnotVolume(self):
    #     pass

    # def testSplitCurve(self):
    #     pass

    # def testSplitSurface(self):
    #     pass

    # def testWindowCurve(self):
    #     pass

    # def testWindowSurface(self):
    #     pass

    # def testReverseCurve(self):
    #     pass

    # def testComputeCurveData(self):
    #     pass

    # def testComputeSurfaceData(self):
    #     pass
