# Standard Python modules
from typing import List, Optional, Tuple, Union

# External modules
import numpy as np
from scipy.sparse import linalg

# Local modules
from . import libspline
from .spline import Spline
from .utils import Error, _assembleMatrix, checkInput, closeTecplot, openTecplot, writeTecplot1D


class BSplineCurve(Spline):
    def __init__(self, degree: int, knotVec: np.ndarray, ctrlPnts: np.ndarray) -> None:
        self.degree = degree
        self.ctrlPnts = ctrlPnts
        self.knotVec = knotVec

        # Attributes we will use in methods
        self.gpts = None  # Greville points vector
        self.sdata = None
        self.u = None  # Parametric coordinate vector

        super(BSplineCurve, self).__init__()

    @property
    def nCtl(self) -> int:
        return self.ctrlPnts.shape[0]

    @property
    def nDim(self) -> int:
        return self.ctrlPnts.shape[1]

    @property
    def degree(self) -> int:
        return self._degree

    @degree.setter
    def degree(self, value: int) -> None:
        self._degree = value

    @property
    def knotVec(self) -> np.ndarray:
        return self._knotVec

    @knotVec.setter
    def knotVec(self, knotVec: np.ndarray) -> None:
        if len(knotVec) == (self.nCtl + self.degree + 1):
            self._knotVec = knotVec
        else:
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(knotVec)} and it should be length {self.nCtl + self._degree + 1}"
            )

    @property
    def ctrlPnts(self) -> np.ndarray:
        return self._ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.ndim >= 2:
            self._ctrlPnts = ctrlPnts
        else:
            raise ValueError(
                "Control point vector must be a 2D array of shape (nCtl, nDim). "
                f"The input control point vector was shape: {ctrlPnts.shape}"
            )

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
        vals = libspline.evalcurve(np.atleast_1d(u), self.knotVec, self.degree, self.ctrlPnts.T)

        return vals.squeeze().T

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """
        Equivalent to getValue()
        """
        return self.getValue(u)

    def calcGrevillePoints(self) -> None:
        """Calculate the Greville points"""
        self.gpts = np.zeros(self.nCtl)
        for i in range(self.nCtl):
            for n in range(self.degree - 1):  # degree loop
                self.gpts[i] += self.knotVec[i + n + 1]
            self.gpts[i] = self.gpts[i] / (self.degree - 1)

    def calcInterpolatedGrevillePoints(self) -> None:
        """
        Compute greville points, but with additional interpolated knots
        """
        self.calcGrevillePoints()
        s = [self.gpts[0]]
        N = 100
        for i in range(len(self.gpts) - 1):
            for j in range(N):
                s.append((self.gpts[i + 1] - self.gpts[i]) * (j + 1) / (N + 1) + self.gpts[i])
            s.append(self.gpts[i + 1])

        self.sdata = np.array(s)


class BSplineSurface(Spline):
    def __init__(self, uDegree: int, vDegree: int, ctrlPnts: np.ndarray, uKnotVec: np.ndarray, vKnotVec: np.ndarray):
        self.uDegree = uDegree
        self.vDegree = vDegree
        self.ctrlPnts = ctrlPnts
        self.uKnotVec = uKnotVec
        self.vKnotVec = vKnotVec

        # Other attributes
        self.u = None
        self.v = None
        self.U = None
        self.V = None
        self.edgeCurves: List[BSplineCurve] = [None, None, None, None]

        super(BSplineSurface, self).__init__()

    @property
    def nCtlu(self) -> int:
        return self.ctrlPnts.shape[0]

    @property
    def nCtlv(self) -> int:
        return self.ctrlPnts.shape[1]

    @property
    def umin(self) -> float:
        return self.uKnotVec[0]

    @property
    def umax(self) -> float:
        return self.uKnotVec[-1]

    @property
    def vmin(self) -> float:
        return self.vKnotVec[0]

    @property
    def vmax(self) -> float:
        return self.vKnotVec[-1]

    @property
    def nDim(self) -> int:
        return self.ctrlPnts.shape[2]

    @property
    def uDegree(self) -> int:
        return self._uDegree

    @uDegree.setter
    def uDegree(self, value: int) -> None:
        self._uDegree = value

    @property
    def vDegree(self) -> int:
        return self._vDegree

    @vDegree.setter
    def vDegree(self, value: int) -> None:
        self._vDegree = value

    @property
    def uKnotVec(self) -> np.ndarray:
        return self._uKnotVec

    @uKnotVec.setter
    def uKnotVec(self, uKnotVec: np.ndarray) -> None:
        if len(uKnotVec) == (self.nCtlu + self.uDegree):
            self._uKnotVec = uKnotVec
        else:
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(uKnotVec)} and it should be length {self.nCtlu + self._uDegree}"
            )

    @property
    def vKnotVec(self) -> np.ndarray:
        return self._vKnotVec

    @vKnotVec.setter
    def vKnotVec(self, vKnotVec: np.ndarray) -> None:
        if len(vKnotVec) == (self.nCtlv + self.vDegree):
            self._vKnotVec = vKnotVec
        else:
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(vKnotVec)} and it should be length {self.nCtlv + self._vDegree}"
            )

    @property
    def ctrlPnts(self) -> np.ndarray:
        return self._ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.ndim >= 2:
            self._ctrlPnts = ctrlPnts
        else:
            raise ValueError(
                "Control point vector must be a 2D array of shape (nCtlu, nCtlv, nDim). "
                f"The input control point vector was shape: {ctrlPnts.shape}"
            )

    def setEdgeCurves(self) -> None:
        """Create curve spline objects for each of the edges"""
        self.edgeCurves[0] = BSplineCurve(degree=self.uDegree, knotVec=self.uKnotVec, ctrlPnts=self.ctrlPnts[:, 0])
        self.edgeCurves[1] = BSplineCurve(degree=self.uDegree, knotVec=self.uKnotVec, ctrlPnts=self.ctrlPnts[:, -1])
        self.edgeCurves[2] = BSplineCurve(degree=self.vDegree, knotVec=self.vKnotVec, ctrlPnts=self.ctrlPnts[0, :])
        self.edgeCurves[3] = BSplineCurve(degree=self.vDegree, knotVec=self.vKnotVec, ctrlPnts=self.ctrlPnts[-1, :])

    def getValueEdge(self, edgeIdx: int, u: np.ndarray) -> np.ndarray:
        curve = self.edgeCurves[edgeIdx]
        return curve(u)

    def getValueCorner(self, corner: int) -> np.ndarray:
        if corner not in [0, 1, 2, 3]:
            raise ValueError("Corner must be in range [0,3]")

        if corner == 0:
            return self.getValue(self.umin, self.vmin)
        elif corner == 1:
            return self.getValue(self.umax, self.vmin)
        elif corner == 2:
            return self.getValue(self.umin, self.vmax)
        elif corner == 3:
            return self.getValue(self.umax, self.vmax)

    def getValue(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the b-spline surface at parametric positions u,v. This is the
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
        vals = libspline.eval_surface(u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPnts.T)
        return vals.squeeze().T

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Equivalant to getValue()
        """
        return self.getValue(u, v)

    def getDerivative(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
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
            Spline derivative evaluation at u,vall points u,v. Shape
            depend on the input.
        """
        if not u.shape == v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        if not np.ndim(u) == 0:
            raise ValueError("'getDerivative' only accepts scalar arguments.")

        deriv = libspline.eval_surface_deriv(
            u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPnts.T
        )
        return deriv.T

    def getSecondDerivative(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the second derivatvies of the spline surface

        deriv = [ (d^2)/(du^2)    (d^2)/(dudv) ]
                [ (d^2)/(dudv)    (d^2)/(dv^2) ]

        Parameters
        ----------
        u : float
            Parametric u value
        v : float
            Parametric v value

        Returns
        -------
        deriv : np.ndarray size (2,2,3)
            Spline derivative evaluation at u,vall points u,v. Shape
            depend on the input.
        """
        if not u.shape == v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        if not np.ndim(u) == 0:
            raise ValueError("'getDerivative' only accepts scalar arguments.")

        deriv = libspline.eval_surface_deriv2(
            u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPnts.T
        )
        return deriv.T

    def getBounds(self) -> Tuple[np.ndarray]:
        """Determine the extents of the surface

        Returns
        -------
        xMin : np.ndarray of length 3
            Lower corner of the bounding box
        xMax : np.ndarray of length 3
            Upper corner of the bounding box
        """
        if self.nDim != 3:
            raise Error("getBounds is only defined for nDim = 3")

        cx = self.ctrlPnts[:, :, 0].flatten()
        cy = self.ctrlPnts[:, :, 1].flatten()
        cz = self.ctrlPnts[:, :, 2].flatten()

        Xmin = np.array([min(cx), min(cy), min(cz)])
        Xmax = np.array([max(cx), max(cy), max(cz)])

        return Xmin, Xmax


class BSplineVolume(Spline):
    def __init__(self):
        super(BSplineVolume, self).__init__()
