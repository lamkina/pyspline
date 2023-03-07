# Standard Python modules
from typing import Optional, Tuple

# External modules
import numpy as np

# Local modules
from . import libspline
from .bspline import BSplineCurve, BSplineSurface


# Define dimension agnostic helper functions for NURBS objects
def combineCtrlPnts(ctrlPnts: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    # Get the shape of the control points
    shape = list(ctrlPnts.shape)
    wShape = shape
    wShape[-1] = 1

    # Check the input weight array and set the value
    # If weights is None we set them to an array of 1's
    if weights is not None:
        if weights.shape[0] != wShape[0]:
            raise ValueError(
                f"The shape of the weight array must be {wShape} "
                "based on the shape of the control point array. "
                f"The current shape is {weights.shape}."
            )
        weights = weights.reshape(wShape)
    else:
        weights = np.ones((wShape))

    # Initialize the weighted control points array
    ctrlPntsW = np.append(ctrlPnts * weights, weights, -1)

    return ctrlPntsW


def separateCtrlPnts(ctrlPntsW: np.ndarray) -> Tuple[np.ndarray]:
    # Convert the shape to a list so we can alter it
    newShape = list(ctrlPntsW.shape)
    newShape[-1] = 3

    # Flatten out the control points into a 2D array of length nDim + 1
    temp = ctrlPntsW.reshape(-1, ctrlPntsW.shape[-1])

    # Get the control points
    ctrlPnts = temp[:, :3].reshape(newShape)

    # Get the weights and flatten the column array
    weights = temp[:, -1].flatten()

    # Divide the control points by the weights and reshape into the correct size
    ctrlPnts = (ctrlPnts.T / weights).T

    # Reshape the control points back to the expected shape
    ctrlPnts = ctrlPnts.reshape(newShape)

    return ctrlPnts, weights


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
        if weights is None:
            weights = np.ones(self.nCtl)
        else:
            if weights.shape != (self.nCtl,):
                raise ValueError(f"Weights must be of shape (nCtl,). Argument was of shape {weights.shape}")
            weights = weights

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
        ctrlPnts = separateCtrlPnts(ctrlPntsW)
        super(NURBSSurface, self).__init__(uDegree, vDegree, ctrlPnts, uKnotVec, vKnotVec)

    @property
    def ctrlPntsW(self) -> np.ndarray:
        return self.ctrlPntsW

    @ctrlPntsW.setter
    def ctrlPntsW(self, ctrlPntsW: np.ndarray) -> None:
        if ctrlPntsW.shape != (self.nCtlu, self.nCtlv, self.nDim + 1):
            raise ValueError(
                "Control points must be of shape (nCtlu, nCtlv, nDim)="
                f"({self.nCtlu, self.nCtlv, self.nDim+1})."
                f"Input was of shape={ctrlPntsW.shape}"
            )
        self.ctrlPntsW = ctrlPntsW

    @property
    def ctrlPnts(self) -> np.ndarray:
        ctrlPnts, _ = separateCtrlPnts(self.ctrlPntsW)
        return ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.shape != (self.nCtlu, self.nCtlv, self.nDim):
            raise ValueError(
                "Control points must be of shape (nCtlu, nCtlv, nDim)="
                f"({self.nCtlu, self.nCtlv, self.nDim})."
                f"Input was of shape={ctrlPnts.shape}"
            )
        self.ctrlPntsW = combineCtrlPnts(ctrlPnts, self.weights)

    @property
    def weights(self) -> np.ndarray:
        _, weights = separateCtrlPnts(self.ctrlPntsW)
        return weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        if weights.shape != (self.nCtlu, self.nCtlv):
            raise ValueError(
                f"Weights must be of shape (nCtlu,nCtlv)=({self.nCtlu, self.nCtlv}). "
                f"Argument was of shape {weights.shape}"
            )
        weights = weights

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

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Equivalant to getValue()
        """
        return self.getValue(u, v)
