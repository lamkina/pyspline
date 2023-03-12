# Standard Python modules
from typing import Tuple

# External modules
import numpy as np


def paramCurve(points: np.ndarray, paramType: str = "arc") -> np.ndarray:
    """Compute the parameterization for a curve.

    Parameters
    ----------
    points : np.ndarray
        The input points. The array should have shape (nPts, nDim), where nPts is the number of points
        and nDim is the dimensionality of each point.
    paramType : str, optional
        The type of parameterization, by default "arc"

    Returns
    -------
    np.ndarray
        The parameteric coordinate array of the curve of shape (nPts,)

    Raises
    ------
    ValueError
        If the parameterization type is not supported.
    """
    if paramType == "arc":
        return chordLengthParam(points)
    elif paramType == "centripetal":
        return centripetalParam(points)
    else:
        raise ValueError(
            f"Argument 'interpType' got an invalid value of {paramType}. Valid values are: ['arc', 'centripetal']"
        )


def chordLengthParam(points: np.ndarray) -> np.ndarray:
    """Chord length based curve parameterization.

    Parameters
    ----------
    points : np.ndarray
        The input points. The array should have shape (nPts, nDim), where nPts is the number of points
        and nDim is the dimensionality of each point.

    Returns
    -------
    np.ndarray
        The parametric coordinates.
    """
    numPts = points.shape[0]
    u = np.zeros(numPts)

    # Calculate the distance between successive points
    distances = np.zeros(numPts + 1)
    distances[1:numPts] = np.linalg.norm(points[1:numPts] - points[0 : numPts - 1], axis=1)
    distances[-1] = 1.0

    # Calculate the total chord length
    distance = np.sum(distances[1:-1])

    u = np.cumsum(distances[:-1]) / distance

    return u


def centripetalParam(points: np.ndarray) -> np.ndarray:
    numPts = points.shape[0]
    u = np.zeros(numPts)

    # Calculate the distance between successive points
    distances = np.zeros(numPts + 1)
    distances[1:numPts] = np.sqrt(np.linalg.norm(points[1:numPts] - points[0 : numPts - 1], axis=1))
    distances[-1] = 1.0

    # Calculate the total chord length
    distance = np.sum(distances[1:-1])

    u = np.cumsum(distances[:-1]) / distance

    return u


def paramSurf(points: np.ndarray) -> Tuple[np.ndarray]:
    """Compute the parametric coordinates for a surface in the u and v
    directions.

    Parameters
    ----------
    points : np.ndarray
        The array of 3D points to fit the surface to, with shape (nu, nv, 3).

    Returns
    -------
    Tuple[np.ndarray]
        The parameteric coordinates in the u and v directions and the
        grid of U,V parametric coordinate pairs for the surface.  The
        tuple is ordered as u, v, U, V
    """
    nu, nv, _ = points.shape

    u = np.zeros(nu, "d")
    U = np.zeros((nu, nv), "d")
    singularSounter = 0

    # Loop over each v, and average the 'u' parameter
    for j in range(nv):
        temp = np.zeros(nu, "d")

        for i in range(nu - 1):
            temp[i + 1] = temp[i] + np.linalg.norm(points[i, j] - points[i + 1, j])

        if temp[-1] == 0:  # Singular point
            singularSounter += 1
            temp[:] = 0.0
            U[:, j] = np.linspace(0, 1, nu)
        else:
            temp = temp / temp[-1]
            U[:, j] = temp.copy()

        u += temp  # accumulate the u-parameter calculations for each j

    u = u / (nv - singularSounter)  # divide by the number of 'j's we had

    v = np.zeros(nv, "d")
    V = np.zeros((nu, nv), "d")
    singularSounter = 0

    # Loop over each u and average the 'v' parameter
    for i in range(nu):
        temp = np.zeros(nv, "d")
        for j in range(nv - 1):
            temp[j + 1] = temp[j] + np.linalg.norm(points[i, j] - points[i, j + 1])

        if temp[-1] == 0:  # Singular point
            singularSounter += 1
            temp[:] = 0.0
            V[i, :] = np.linspace(0, 1, nv)
        else:
            temp = temp / temp[-1]
            V[i, :] = temp.copy()

        v += temp  # accumulate the v-parameter calculations for each i

    v = v / (nu - singularSounter)  # divide by the number of 'i's we had

    return u, v, U, V
