# Standard Python modules
from typing import List, Tuple, Union

# External modules
import numpy as np

# Local modules
from . import libspline, utils
from .spline import Spline


class BSplineCurve(Spline):
    def __init__(self, degree: int, knotVec: np.ndarray, ctrlPnts: np.ndarray) -> None:
        self.degree = degree
        self.ctrlPnts = ctrlPnts
        self.knotVec = knotVec

        # Attributes we will use in methods
        self.gPts = None  # Greville points vector
        self.uData = None  # Interpolated points vector
        self.data = None
        self.u = None  # Parametric coordinate vector

        super(BSplineCurve, self).__init__()

    @property
    def pDim(self) -> int:
        return 1

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
        kv = self._knotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @knotVec.setter
    def knotVec(self, knotVec: np.ndarray) -> None:
        # Check the dimension and make sure the knots are ascending
        if not len(knotVec) == (self.nCtl + self.degree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(knotVec)} and it should be length {self.nCtl + self._degree + 1}"
            )

        if not np.all(knotVec[:-1] <= knotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        self._knotVec = knotVec

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

    def getValue(self, u: Union[float, np.ndarray]) -> np.ndarray:
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
        vals = libspline.evalcurve(np.atleast_1d(u), self.knotVec, self.degree, self.ctrlPnts.T)

        return vals.squeeze().T

    def getDerivative(self, u: float, order: int) -> np.ndarray:
        ck = libspline.derivevalcurve(u, self.knotVec, self.degree, self.ctrlPnts.T, order)

        return ck.T

    def __call__(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Equivalent to getValue()
        """
        return self.getValue(u)

    def computeData(self, recompute: bool = False) -> None:
        if self.data is None or recompute:
            self.gPts = utils.calculateGrevillePoints(self.degree, self.nCtl, self.knotVec)
            self.uData = utils.calculateInterpolatedGrevillePoints(10, self.gPts)
            self.data = self.getValue(self.uData)


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
        self.uData = None
        self.vData = None
        self.data = None

        self.edgeCurves: List[BSplineCurve] = [None, None, None, None]
        self.setEdgeCurves()

        super(BSplineSurface, self).__init__()

    @property
    def pDim(self) -> int:
        return 2

    @property
    def nCtlu(self) -> int:
        return self.ctrlPnts.shape[0]

    @property
    def nCtlv(self) -> int:
        return self.ctrlPnts.shape[1]

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
        kv = self._uKnotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @uKnotVec.setter
    def uKnotVec(self, uKnotVec: np.ndarray) -> None:
        if not np.all(uKnotVec[:-1] <= uKnotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        if len(uKnotVec) != (self.nCtlu + self.uDegree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(uKnotVec)} and it should be length {self.nCtlu + self._uDegree + 1}"
            )

        self._uKnotVec = uKnotVec

    @property
    def vKnotVec(self) -> np.ndarray:
        kv = self._vKnotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @vKnotVec.setter
    def vKnotVec(self, vKnotVec: np.ndarray) -> None:
        if not np.all(vKnotVec[:-1] <= vKnotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        if len(vKnotVec) != (self.nCtlv + self.vDegree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(vKnotVec)} and it should be length {self.nCtlv + self._vDegree + 1}"
            )

        self._vKnotVec = vKnotVec

    @property
    def ctrlPnts(self) -> np.ndarray:
        return self._ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.ndim == 3:
            self._ctrlPnts = ctrlPnts
        else:
            raise ValueError(
                "Control point vector must be a 3D array of shape (nCtlu, nCtlv, nDim). "
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
            return self.getValue(0, 0)
        elif corner == 1:
            return self.getValue(1, 0)
        elif corner == 2:
            return self.getValue(0, 1)
        elif corner == 3:
            return self.getValue(1, 1)

    def getValue(self, u: Union[float, np.ndarray], v: Union[float, np.ndarray]) -> np.ndarray:
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

        if u.shape != v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        vals = libspline.evalsurface(u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPnts.T)
        return vals.squeeze().T

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Equivalant to getValue()
        """
        return self.getValue(u, v)

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
            Spline derivative evaluation at u,vall points u,v. Shape
            depend on the input.
        """
        if not u.shape == v.shape:
            raise ValueError(f"u and v must have the same shape.  u has shape {u.shape} and v has shape {v.shape}.")

        if not np.ndim(u) == 0:
            raise ValueError("'getDerivative' only accepts scalar arguments.")

        deriv = libspline.derivevalsurface(
            u, v, self.uKnotVec, self.vKnotVec, self.uDegree, self.vDegree, self.ctrlPnts.T, order
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
            raise ValueError("getBounds is only defined for nDim = 3")

        cx = self.ctrlPnts[:, :, 0].flatten()
        cy = self.ctrlPnts[:, :, 1].flatten()
        cz = self.ctrlPnts[:, :, 2].flatten()

        Xmin = np.array([min(cx), min(cy), min(cz)])
        Xmax = np.array([max(cx), max(cy), max(cz)])

        return Xmin, Xmax

    def computeData(self, recompute: bool = False) -> None:
        if self.data is None or recompute:
            curve0 = self.edgeCurves[0]
            gPts0 = utils.calculateGrevillePoints(curve0.degree, curve0.nCtl, curve0.knotVec)

            curve2 = self.edgeCurves[2]
            gPts2 = utils.calculateGrevillePoints(curve2.degree, curve2.nCtl, curve2.knotVec)

            self.uData = utils.calculateInterpolatedGrevillePoints(10, gPts0)
            self.vData = utils.calculateInterpolatedGrevillePoints(10, gPts2)

            self.V, self.U = np.meshgrid(self.vData, self.uData)
            self.data = self.getValue(self.U, self.V)


class BSplineVolume(Spline):
    def __init__(
        self,
        uDegree: int,
        vDegree: int,
        wDegree: int,
        ctrlPnts: np.ndarray,
        uKnotVec: np.ndarray,
        vKnotVec: np.ndarray,
        wKnotVec: np.ndarray,
    ) -> None:
        self.uDegree = uDegree
        self.vDegree = vDegree
        self.wDegree = wDegree
        self.ctrlPnts = ctrlPnts
        self.uKnotVec = uKnotVec
        self.vKnotVec = vKnotVec
        self.wKnotVec = wKnotVec

        # Other attributes
        self.u = None
        self.v = None
        self.U = None
        self.V = None
        self.uData = None
        self.vData = None
        self.data = None

        super(BSplineVolume, self).__init__()

    @property
    def pDim(self) -> int:
        return 3

    @property
    def nCtlu(self) -> int:
        return self.ctrlPnts.shape[0]

    @property
    def nCtlv(self) -> int:
        return self.ctrlPnts.shape[1]

    @property
    def nCtlw(self) -> int:
        return self.ctrlPnts.shape[2]

    @property
    def nDim(self) -> int:
        return self.ctrlPnts.shape[3]

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
    def uDegree(self, value: int) -> None:
        self._vDegree = value

    @property
    def wDegree(self) -> int:
        return self._wDegree

    @wDegree.setter
    def uDegree(self, value: int) -> None:
        self._wDegree = value

    @property
    def uKnotVec(self) -> np.ndarray:
        kv = self._uKnotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @uKnotVec.setter
    def uKnotVec(self, uKnotVec: np.ndarray) -> None:
        if not np.all(uKnotVec[:-1] <= uKnotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        if len(uKnotVec) != (self.nCtlu + self.uDegree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(uKnotVec)} and it should be length {self.nCtlu + self._uDegree + 1}"
            )

        self._uKnotVec = uKnotVec

    @property
    def vKnotVec(self) -> np.ndarray:
        kv = self._vKnotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @vKnotVec.setter
    def vKnotVec(self, vKnotVec: np.ndarray) -> None:
        if not np.all(vKnotVec[:-1] <= vKnotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        if len(vKnotVec) != (self.nCtlv + self.vDegree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(vKnotVec)} and it should be length {self.nCtlv + self._vDegree + 1}"
            )

        self._vKnotVec = vKnotVec

    @property
    def wKnotVec(self) -> np.ndarray:
        kv = self._wKnotVec
        return (kv - kv[0]) / (kv[-1] - kv[0])

    @wKnotVec.setter
    def wKnotVec(self, wKnotVec: np.ndarray) -> None:
        if not np.all(wKnotVec[:-1] <= wKnotVec[1:]):
            raise ValueError("Knot vector is not in ascending order.")

        if len(wKnotVec) != (self.nCtlw + self.vDegree + 1):
            raise ValueError(
                f"Knot vector is not the correct length. "
                f"Input length was {len(wKnotVec)} and it should be length {self.nCtlv + self._wDegree + 1}"
            )

        self._wKnotVec = wKnotVec

    @property
    def ctrlPnts(self) -> np.ndarray:
        return self._ctrlPnts

    @ctrlPnts.setter
    def ctrlPnts(self, ctrlPnts: np.ndarray) -> None:
        if ctrlPnts.ndim == 3:
            self._ctrlPnts = ctrlPnts
        else:
            raise ValueError(
                "Control point vector must be a 3D array of shape (nCtlu, nCtlv, nDim). "
                f"The input control point vector was shape: {ctrlPnts.shape}"
            )

    def setEdgeSurfaces(self):
        pass

    def setEdgeCurves(self):
        pass

    def getValueCorner(self, corner: int) -> float:
        if corner not in range(0, 8):
            raise ValueError("Corner must be in range [0,7] inclusive.")

        pass

    def getValueEdge(self, edge, u):
        pass

    def getBounds(self):
        pass

    def getValue(
        self, u: Union[float, np.ndarray], v: Union[float, np.ndarray], w: Union[float, np.ndarray]
    ) -> np.ndarray:
        u = np.atleast_3d(u).T
        v = np.atleast_3d(v).T
        w = np.atleast_3d(w).T

        if not u.shape == v.shape == w.shape:
            raise ValueError("u, v, and w must have the same shape.")

        vals = libspline.evalvolume(
            u,
            v,
            w,
            self.uKnotVec,
            self.vKnotVec,
            self.wKnotVec,
            self.uDegree,
            self.vDegree,
            self.wDegree,
            self.ctrlPnts.T,
        )

        return vals.squeeze().T

    def __call__(
        self, u: Union[float, np.ndarray], v: Union[float, np.ndarray], w: Union[float, np.ndarray]
    ) -> np.ndarray:
        return self.getValue(u, v, w)

    def computeData(self, recompute=False):
        pass
