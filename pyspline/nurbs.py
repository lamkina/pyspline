# Standard Python modules
from typing import List, Union

# External modules
import numpy as np

# Local modules
from . import libspline, utils
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
    def nDim(self) -> int:
        return self.ctrlPnts.shape[1]

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

    def getValue(self, u: Union[np.ndarray, float]) -> np.ndarray:
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

        u = np.array(u).T

        vals, _ = libspline.evalcurvenurbs(np.atleast_1d(u), self.knotVec, self.degree, self.ctrlPntsW.T)

        return vals.squeeze().T[:3] if np.ndim(vals.squeeze().T) == 1 else vals.squeeze().T[:, : self.nDim]

    def getWeight(self, u: Union[np.ndarray, float]) -> np.ndarray:
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

        u = np.array(u).T

        _, weights = libspline.evalcurvenurbs(np.atleast_1d(u), self.knotVec, self.degree, self.ctrlPntsW.T)

        return weights.T

    def __call__(self, u: Union[np.ndarray, float]) -> np.ndarray:
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

    def getDerivative(self, u: float, order: int) -> np.ndarray:
        ck = libspline.derivevalcurvenurbs(u, self.knotVec, self.degree, self.ctrlPntsW.T, order)
        return ck.T

    def computeData(self, recompute: bool = False) -> None:
        if self.data is None or recompute:
            self.gPts = utils.calculateGrevillePoints(self.degree, self.nCtl, self.knotVec)
            self.uData = utils.calculateInterpolatedGrevillePoints(10, self.gPts)
            self.data = self.getValue(self.uData)


class NURBSSurface(BSplineSurface):
    def __init__(
        self, uDegree: int, vDegree: int, ctrlPntsW: np.ndarray, uKnotVec: np.ndarray, vKnotVec: np.ndarray
    ) -> None:
        # Initialize the NURBS specific attributes
        self.ctrlPntsW = ctrlPntsW

        # Initialize the baseclass BSpline object
        super(NURBSSurface, self).__init__(uDegree, vDegree, self.ctrlPnts, uKnotVec, vKnotVec)

        self._rational = True
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
        return vals.squeeze().T

    def getDerivative(self, u: np.ndarray, v: np.ndarray, order) -> np.ndarray:
        """Evaluate the first derivatvies of the spline surface

        Parameters
        ----------
        u : np.ndarray
            Parametric u value
        v : np.ndarray
            Parametric v value

        Returns
        -------
        deriv : np.ndarray size (2,3)
            Spline derivative evaluation at u,v. Shape depends on the
            input.
        """
        if not u.shape == v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        if not np.ndim(u) == 0:
            raise ValueError("'getDerivative' only accepts scalar arguments.")

        deriv = libspline.derivevalsurfacenurbs(
            u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPntsW.T, order
        )

        return deriv.T

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

    def computeData(self, recompute: bool = False) -> None:
        if self.data is None or recompute:
            curve0 = self.edgeCurves[0]
            gPts0 = utils.calculateGrevillePoints(curve0.degree, curve0.nCtl, curve0.knotVec)

            curve2 = self.edgeCurves[2]
            gPts2 = utils.calculateGrevillePoints(curve2.degree, curve2.nCtl, curve2.knotVec)

            self.uData = utils.calculateInterpolatedGrevillePoints(3, gPts0)
            self.vData = utils.calculateInterpolatedGrevillePoints(3, gPts2)

            self.V, self.U = np.meshgrid(self.vData, self.uData)
            self.data = self.getValue(self.U, self.V)[:, :, :-1]


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
