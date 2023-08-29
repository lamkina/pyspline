# Standard Python modules
import unittest

# External modules
import numpy as np
from numpy.testing import assert_allclose

# First party modules
from pyspline.bspline import BSplineCurve
from pyspline.nurbs import NURBSCurve


class TestBSplineCurve(unittest.TestCase):
    N_PROCS = 1

    @classmethod
    def setUpClass(self):
        # Create a 2D BSpline
        self.knotVec = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
        self.degree = 3
        self.ctrlPnts2d = np.array([[0.0, 0.0], [1.0, 1.0], [1.5, 1.5], [2.0, 4.0], [3.0, 4.0], [3.5, 3.0], [4.0, 1.0]])
        self.curve2d = BSplineCurve(self.degree, self.knotVec, self.ctrlPnts2d)

        # Create a 3D BSpline (use same knot vector and degree for simplicity)
        self.ctrlPnts3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.0],
                [2.0, 4.0, 1.5],
                [3.0, 4.0, 2.5],
                [3.5, 3.0, 3.0],
                [4.0, 1.0, 2.5],
            ]
        )
        self.curve3d = BSplineCurve(self.degree, self.knotVec, self.ctrlPnts3d)

        # Create a curve with a non normalized knot vector to test the normalization
        self.knotVecNonNorm = np.array([0.0, 0.0, 0.0, 1.0, 2.5, 3.0, 3.0, 3.0])
        ctrlPnts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.curveNonNorm = BSplineCurve(2, self.knotVecNonNorm, ctrlPnts)

    def testDegree(self):
        self.assertEqual(self.curve2d.degree, 3)
        self.assertEqual(self.curve3d.degree, 3)

    def testNDim(self):
        self.assertEqual(self.curve2d.nDim, 2)
        self.assertEqual(self.curve3d.nDim, 3)

    def testPDim(self):
        self.assertEqual(self.curve2d.pDim, 1)
        self.assertEqual(self.curve3d.pDim, 1)

    def testRational(self):
        self.assertFalse(self.curve2d.rational)
        self.assertFalse(self.curve3d.rational)

    def testKnotVec(self):
        assert_allclose(self.curve2d.knotVec, self.knotVec)
        assert_allclose(self.curve3d.knotVec, self.knotVec)

        # Calculate the normalized knot vector
        kv = self.knotVecNonNorm
        kvNorm = (kv - kv[0]) / (kv[-1] - kv[0])

        assert_allclose(self.curveNonNorm.knotVec, kvNorm)

    def testCtrlPnts(self):
        assert_allclose(self.curve2d.ctrlPnts, self.ctrlPnts2d)
        assert_allclose(self.curve3d.ctrlPnts, self.ctrlPnts3d)

    def testSetKnotVec(self):
        # Test that an error is thrown if the knot vector is the wrong size
        with self.assertRaises(ValueError):
            self.curve2d.knotVec = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        with self.assertRaises(ValueError):
            self.curve3d.knotVec = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        # Test that an error is thrown if the knot vector is not ascending
        with self.assertRaises(ValueError):
            self.curve2d.knotVec = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.125, 0.0, 1.0, 1.0, 1.0, 1.0])

        with self.assertRaises(ValueError):
            self.curve3d.knotVec = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.125, 0.0, 1.0, 1.0, 1.0, 1.0])

    def testSetCtrlPnts(self):
        # Test that setting the control points with the wrong dimension raises an error
        with self.assertRaises(ValueError):
            self.curve2d.ctrlPnts = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0])


class TestNURBSCurve(unittest.TestCase):
    N_PROCS = 1

    @classmethod
    def setUpClass(self) -> None:
        # Create a degree 2 nurbs circle of radius 1
        degree = 2
        self.knotVec = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])
        self.ctrlPntsW = np.array(
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
        self.nurbsCircle = NURBSCurve(degree, self.knotVec, self.ctrlPntsW)

    def testDegree(self):
        self.assertEqual(self.nurbsCircle.degree, 2)

    def testNDim(self):
        self.assertEqual(self.nurbsCircle.nDim, 3)

    def testPDim(self):
        self.assertEqual(self.nurbsCircle.pDim, 1)

    def testKnotVec(self):
        assert_allclose(self.nurbsCircle.knotVec, self.knotVec)

    def testCtrlPnts(self):
        ctrlPnts = np.divide(self.ctrlPntsW[:, :-1].T, self.ctrlPntsW[:, -1]).T
        assert_allclose(self.nurbsCircle.ctrlPnts, ctrlPnts)

    def testCtrlPntsW(self):
        assert_allclose(self.nurbsCircle.ctrlPntsW, self.ctrlPntsW)

    def testSetCtrlPntsW(self):
        ctrlPtnsWCopy = self.ctrlPntsW.copy()
        ctrlPtnsWCopy[:, -1] = 0.0

        # Test that setting the weighted control points with zero weights
        # retuns the correct error
        with self.assertRaises(ValueError):
            self.nurbsCircle.ctrlPntsW = ctrlPtnsWCopy

    def testRational(self):
        self.assertTrue(self.nurbsCircle.rational)

    def testSetWeights(self):
        # Check that setting weights with the wrong shape array fails
        with self.assertRaises(ValueError):
            self.nurbsCircle.weights = np.ones(4)

        with self.assertRaises(ValueError):
            self.nurbsCircle.weights = np.ones((10, 1))

        # Test setting weights as zero raises a value error
        with self.assertRaises(ValueError):
            self.nurbsCircle.weights = np.zeros(self.nurbsCircle.nCtl)
