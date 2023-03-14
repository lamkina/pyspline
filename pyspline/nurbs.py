# Standard Python modules
from typing import List

# External modules
import numpy as np

# Local modules
from . import libspline
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume
from .compatibility import combineCtrlPnts, separateCtrlPnts


class NURBSCurve(BSplineCurve):
    def __init__(self, degree: int, knotVec: np.ndarray, ctrlPntsW: np.ndarray) -> None:
        # Initialize the NURBS specific attributes
        self.ctrlPntsW = ctrlPntsW

        # Initialize the baseclass BSpline object
        super(NURBSCurve, self).__init__(degree, knotVec, self.ctrlPnts)
        self._rational = True

    @property
    def ctrlPntsW(self) -> np.ndarray:
        return self._ctrlPntsW

    @ctrlPntsW.setter
    def ctrlPntsW(self, val: np.ndarray) -> None:
        if np.any(val[:, -1] == 0):
            raise ValueError("Weights cannot have zero entries")

        self._ctrlPntsW = val

    @property
    def ctrlPnts(self) -> np.ndarray:
        ctrlPnts, _ = separateCtrlPnts(self.ctrlPntsW)
        return ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        self.ctrlPntsW = combineCtrlPnts(ctrlPnts, self.weights)

    @property
    def weights(self) -> np.ndarray:
        _, weights = separateCtrlPnts(self.ctrlPntsW)
        return weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        if weights.shape != (self.nCtl,):
            raise ValueError(f"Weights must be of shape (nCtl,). Argument was of shape {weights.shape}")

        if any(weights == 0.0):
            raise ValueError("Weights cannot have zero entries.")

        self.ctrlPntsW = combineCtrlPnts(self.ctrlPnts, weights)

    def getValue(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate the spline at parametric position, u

        Parameters
        ----------
        u : float or np.ndarray
            Parametric position(s) at which to evaluate the curve.

        Returns
        -------
        values : np.ndarray of size nDim or size (N, 3)
            The evaluated points. If a single scalar 'u' was given, the
            result with be an array of length nDim (or a scalar if
            nDim=1). If a vector of 'u' values were given it will be an
            array of size (N, 3) (or size (N) if ndim=1)
        """

        u = u.T

        vals = libspline.evalcurvenurbs(np.atleast_1d(u), self.knotVec, self.degree, self.ctrlPntsW.T)

        return vals.squeeze().T[:, :3]

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Equivalent to `getValue()`

        Parameters
        ----------
        u : np.ndarray
            Parametric position(s) at which to evaluate the curve.

        Returns
        -------
        values : np.ndarray of size nDim or size (N, 3)
            The evaluated points. If a single scalar 'u' was given, the
            result with be an array of length nDim (or a scalar if
            nDim=1). If a vector of 'u' values were given it will be an
            array of size (N, 3) (or size (N) if ndim=1)
        """
        return self.getValue(u)


class NURBSSurface(BSplineSurface):
    def __init__(
        self, uDegree: int, vDegree: int, ctrlPntsW: np.ndarray, uKnotVec: np.ndarray, vKnotVec: np.ndarray
    ) -> None:
        # Initialize the NURBS specific attributes
        self.ctrlPntsW = ctrlPntsW
        self._rational = True

        # Initialize the baseclass BSpline object
        super(NURBSSurface, self).__init__(uDegree, vDegree, self.ctrlPnts, uKnotVec, vKnotVec)

        self.edgeCurves: List[NURBSCurve] = [None, None, None, None]
        self.setEdgeCurves()

    @property
    def ctrlPntsW(self) -> np.ndarray:
        return self._ctrlPntsW

    @ctrlPntsW.setter
    def ctrlPntsW(self, ctrlPntsW: np.ndarray) -> None:
        if np.any(ctrlPntsW[:, :, -1] == 0):
            raise ValueError("Weights cannot have zero entries")
        self._ctrlPntsW = ctrlPntsW

    @property
    def ctrlPnts(self) -> np.ndarray:
        ctrlPnts, _ = separateCtrlPnts(self.ctrlPntsW)
        return ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.ndim != 3:
            raise ValueError(
                "Control point vector must be a 2D array of shape (nCtlu, nCtlv, nDim). "
                f"The input control point vector was shape: {ctrlPnts.shape}"
            )

        self.ctrlPntsW = combineCtrlPnts(ctrlPnts, self.weights)

    @property
    def weights(self) -> np.ndarray:
        _, weights = separateCtrlPnts(self.ctrlPntsW)
        return weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        if weights.shape != (self.nCtlu * self.nCtlv,):
            raise ValueError(
                f"Weights must be of shape (nCtlu * nCtlv,)=({self.nCtlu * self.nCtlv},). "
                f"Argument was of shape {weights.shape}"
            )

        self.ctrlPntsW = combineCtrlPnts(self.ctrlPnts, weights)

    def getValue(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the NURBS surface at parametric positions u,v. This is the
        main function for spline evaluation.

        Parameters
        ----------
        u : float, array or matrix (rank 0, 1, or 2)
            Parametric u values
        v : float, array or matrix (rank 0, 1, or 2)
            Parametric v values

        Returns
        -------
        values : np.ndarray
            Spline evaluation at all points u,v. Shape depend on the
            input. If u,v are scalars, values is array of size nDim. If
            u,v are a 1D list, return is (N,nDim) etc.
        """
        u = u.T
        v = v.T

        if not u.shape == v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        u = np.atleast_2d(u)
        v = np.atleast_2d(v)

        vals = libspline.evalsurfacenurbs(
            u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPntsW.T
        )
        return vals.squeeze().T[:, :, :-1]

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Equivalant to getValue()
        """
        return self.getValue(u, v)

    def setEdgeCurves(self) -> None:
        """Create curve spline objects for each of the edges"""
        self.edgeCurves[0] = NURBSCurve(degree=self.uDegree, knotVec=self.uKnotVec, ctrlPntsW=self.ctrlPntsW[:, 0])
        self.edgeCurves[1] = NURBSCurve(degree=self.uDegree, knotVec=self.uKnotVec, ctrlPntsW=self.ctrlPntsW[:, -1])
        self.edgeCurves[2] = NURBSCurve(degree=self.vDegree, knotVec=self.vKnotVec, ctrlPntsW=self.ctrlPntsW[0, :])
        self.edgeCurves[3] = NURBSCurve(degree=self.vDegree, knotVec=self.vKnotVec, ctrlPntsW=self.ctrlPntsW[-1, :])


class NURBSVolume(BSplineVolume):
    def __init__(
        self,
        uDegree: int,
        vDegree: int,
        wDegree: int,
        ctrlPntsW: np.ndarray,
        uKnotVec: np.ndarray,
        vKnotVec: np.ndarray,
        wKnotVec: np.ndarray,
    ) -> None:
        # Initialize the NURBS specific attributes
        self.ctrlPntsW = ctrlPntsW
        self._rational = True

        # Initialize the baseclass BSpline object
        ctrlPnts = separateCtrlPnts(ctrlPntsW)
        super(NURBSVolume, self).__init__(uDegree, vDegree, wDegree, ctrlPnts, uKnotVec, vKnotVec, wKnotVec)
