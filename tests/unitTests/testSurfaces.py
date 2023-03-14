# Standard Python modules
import unittest

# External modules
import numpy as np
from numpy.testing import assert_allclose

# First party modules
from pyspline import BSplineCurve, BSplineSurface, NURBSCurve, NURBSSurface
from pyspline.export import writeTecplot


class TestBSplineSurface(unittest.TestCase):
    N_PROCS = 1

    @classmethod
    def setUpClass(self):
        # Create a surface using Ex. 3.8 from The NURBS Book
        self.uKnotVec = np.array([0, 0, 0, 1 / 2, 1, 1, 1])
        self.vKnotVec = np.array([0, 0, 0, 1, 1, 1])
        self.uDegree = 2
        self.vDegree = 2
        self.ctrlPnts = np.zeros((4, 3, 3))
        self.ctrlPnts[0, 0] = np.array([0, 0, 0])
        self.ctrlPnts[0, 1] = np.array([0, 2, 2])
        self.ctrlPnts[0, 2] = np.array([0, 4, 0])
        self.ctrlPnts[1, 0] = np.array([3, 0, 3])
        self.ctrlPnts[1, 1] = np.array([3, 2, 5])
        self.ctrlPnts[1, 2] = np.array([3, 4, 3])
        self.ctrlPnts[2, 0] = np.array([6, 0, 3])
        self.ctrlPnts[2, 1] = np.array([6, 2, 5])
        self.ctrlPnts[2, 2] = np.array([6, 4, 3])
        self.ctrlPnts[3, 0] = np.array([9, 0, 0])
        self.ctrlPnts[3, 1] = np.array([9, 2, 2])
        self.ctrlPnts[3, 2] = np.array([9, 4, 0])

        self.surface = BSplineSurface(self.uDegree, self.vDegree, self.ctrlPnts, self.uKnotVec, self.vKnotVec)

    def testuDegree(self):
        self.assertEqual(self.surface.uDegree, self.uDegree)

    def testvDegree(self):
        self.assertEqual(self.surface.vDegree, self.vDegree)

    def testnDim(self):
        self.assertEqual(self.surface.nDim, 3)

    def testnCtlu(self):
        self.assertEqual(self.surface.nCtlu, self.ctrlPnts.shape[0])

    def testnCtlv(self):
        self.assertEqual(self.surface.nCtlv, self.ctrlPnts.shape[1])

    def testpDim(self):
        self.assertEqual(self.surface.pDim, 2)

    def testuKnotVec(self):
        assert_allclose(self.surface.uKnotVec, self.uKnotVec)

        # Test knot vector of the wrong size
        with self.assertRaises(ValueError):
            self.surface.uKnotVec = np.array([0, 0, 0, 1 / 2, 1 / 2, 1, 1, 1])

        # Test non-ascending knot vector
        with self.assertRaises(ValueError):
            self.surface.uKnotVec = np.array([0, 0, 0, 1 / 2, 0, 1, 1])

    def testvKnotVec(self):
        assert_allclose(self.surface.vKnotVec, self.vKnotVec)

        # Test knot vector of the wrong size
        with self.assertRaises(ValueError):
            self.surface.vKnotVec = np.array([0, 0, 0, 1, 1, 1, 1])

        # Test non-ascending knot vector
        with self.assertRaises(ValueError):
            self.surface.vKnotVec = np.array([0, 0, 0, 1, 1 / 2, 1])

    def testctrlPnts(self):
        assert_allclose(self.surface.ctrlPnts, self.ctrlPnts)

        # Test control points of the wrong size
        # Just pass a dummy array of zeros to trigger the error
        with self.assertRaises(ValueError):
            self.surface.ctrlPnts = np.zeros(10)

    def testEdgeCurves(self):
        # Check the curve class and degree
        for curve in self.surface.edgeCurves:
            self.assertIsInstance(curve, BSplineCurve)
            self.assertEqual(curve.degree, 2)

        # Check the curve knot vectors
        assert_allclose(self.surface.edgeCurves[0].knotVec, self.uKnotVec)
        assert_allclose(self.surface.edgeCurves[1].knotVec, self.uKnotVec)
        assert_allclose(self.surface.edgeCurves[2].knotVec, self.vKnotVec)
        assert_allclose(self.surface.edgeCurves[3].knotVec, self.vKnotVec)

        # Check the curve control points
        assert_allclose(self.surface.edgeCurves[0].ctrlPnts, self.ctrlPnts[:, 0])
        assert_allclose(self.surface.edgeCurves[1].ctrlPnts, self.ctrlPnts[:, -1])
        assert_allclose(self.surface.edgeCurves[2].ctrlPnts, self.ctrlPnts[0, :])
        assert_allclose(self.surface.edgeCurves[3].ctrlPnts, self.ctrlPnts[-1, :])

    def testGetBounds(self):
        bounds = self.surface.getBounds()
        assert_allclose(bounds[0], np.array([0.0, 0.0, 0.0]))
        assert_allclose(bounds[1], np.array([9.0, 4.0, 5.0]))


class TestNURBSSurface(unittest.TestCase):
    N_PROCS = 1

    @classmethod
    def setUpClass(self):
        # Create a surface using Ex. 4.3 from The NURBS Book
        self.uKnotVec = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])
        self.vKnotVec = np.array([0, 0, 1 / 2, 1, 1])
        self.uDegree = 2
        self.vDegree = 1
        self.ctrlPntsW = np.zeros((9, 3, 4))
        self.ctrlPntsW[:, 0] = np.array(
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

        self.ctrlPntsW[:, 1] = np.array(
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

        self.ctrlPntsW[:, 2] = np.array(
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

        shape = list(self.ctrlPntsW.shape)
        shape[-1] -= 1
        temp = self.ctrlPntsW.reshape((-1, self.ctrlPntsW.shape[-1]))
        temp = temp[:, :-1]
        ctrlPnts = np.divide(temp.T, self.ctrlPntsW[:, :, -1].flatten()).T
        self.ctrlPnts = ctrlPnts.reshape(shape)

        self.surface = NURBSSurface(self.uDegree, self.vDegree, self.ctrlPntsW, self.uKnotVec, self.vKnotVec)

    def testctrlPntsW(self):
        assert_allclose(self.surface.ctrlPntsW, self.ctrlPntsW)

    def testuDegree(self):
        self.assertEqual(self.surface.uDegree, self.uDegree)

    def testvDegree(self):
        self.assertEqual(self.surface.vDegree, self.vDegree)

    def testnDim(self):
        self.assertEqual(self.surface.nDim, 3)

    def testnCtlu(self):
        self.assertEqual(self.surface.nCtlu, self.ctrlPntsW.shape[0])

    def testnCtlv(self):
        self.assertEqual(self.surface.nCtlv, self.ctrlPntsW.shape[1])

    def testpDim(self):
        self.assertEqual(self.surface.pDim, 2)

    def testuKnotVec(self):
        assert_allclose(self.surface.uKnotVec, self.uKnotVec)

        # Test knot vector of the wrong size
        with self.assertRaises(ValueError):
            self.surface.uKnotVec = np.array([0, 0, 0, 1 / 2, 1 / 2, 1, 1, 1])

        # Test non-ascending knot vector
        with self.assertRaises(ValueError):
            self.surface.uKnotVec = np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.1, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0])

    def testvKnotVec(self):
        assert_allclose(self.surface.vKnotVec, self.vKnotVec)

        # Test knot vector of the wrong size
        with self.assertRaises(ValueError):
            self.surface.vKnotVec = np.array([0, 0, 1, 1])

        # Test non-ascending knot vector
        with self.assertRaises(ValueError):
            self.surface.vKnotVec = np.array([0, 0, 1 / 2, 0, 1])

    def testctrlPnts(self):

        assert_allclose(self.surface.ctrlPnts, self.ctrlPnts)

        # Test control points of the wrong size
        # Just pass a dummy array of zeros to trigger the error
        with self.assertRaises(ValueError):
            self.surface.ctrlPnts = np.zeros((4, 4, 4))

    def testEdgeCurves(self):
        # Check the curve class and degree
        for curve in self.surface.edgeCurves:
            self.assertIsInstance(curve, NURBSCurve)

        self.assertEqual(self.surface.edgeCurves[0].degree, 2)
        self.assertEqual(self.surface.edgeCurves[1].degree, 2)
        self.assertEqual(self.surface.edgeCurves[2].degree, 1)
        self.assertEqual(self.surface.edgeCurves[3].degree, 1)

        # Check the curve knot vectors
        assert_allclose(self.surface.edgeCurves[0].knotVec, self.uKnotVec)
        assert_allclose(self.surface.edgeCurves[1].knotVec, self.uKnotVec)
        assert_allclose(self.surface.edgeCurves[2].knotVec, self.vKnotVec)
        assert_allclose(self.surface.edgeCurves[3].knotVec, self.vKnotVec)

        # Check the curve control points
        assert_allclose(self.surface.edgeCurves[0].ctrlPnts, self.ctrlPnts[:, 0])
        assert_allclose(self.surface.edgeCurves[1].ctrlPnts, self.ctrlPnts[:, -1])
        assert_allclose(self.surface.edgeCurves[2].ctrlPnts, self.ctrlPnts[0, :])
        assert_allclose(self.surface.edgeCurves[3].ctrlPnts, self.ctrlPnts[-1, :])

        # Check the curve weighted control points
        assert_allclose(self.surface.edgeCurves[0].ctrlPntsW, self.ctrlPntsW[:, 0])
        assert_allclose(self.surface.edgeCurves[1].ctrlPntsW, self.ctrlPntsW[:, -1])
        assert_allclose(self.surface.edgeCurves[2].ctrlPntsW, self.ctrlPntsW[0, :])
        assert_allclose(self.surface.edgeCurves[3].ctrlPntsW, self.ctrlPntsW[-1, :])

    def testGetBounds(self):
        bounds = self.surface.getBounds()
        assert_allclose(bounds[0], np.array([-1.0, -1.0, 0.0]))
        assert_allclose(bounds[1], np.array([1.0, 1.0, 4.0]))
