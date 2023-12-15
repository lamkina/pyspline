# Standard Python modules
from copy import deepcopy

# External modules
import numpy as np

# First party modules
from pyspline.customTypes import GEOTYPE


def scale(geo: GEOTYPE, scale: float, inplace: bool = False) -> GEOTYPE:
    if inplace:
        geo.ctrlPnts = geo.ctrlPnts * scale
        return geo
    else:
        geoScaled = deepcopy(geo)
        geoScaled.ctrlPnts = geoScaled.ctrlPnts * scale
        return geoScaled


def rotate(geo: GEOTYPE, axis: np.ndarray, theta: float, inplace: bool = False) -> GEOTYPE:
    # Normalize the axis incase the user forgot
    axis = np.array(axis) / np.linalg.norm(axis)

    # Compute the cross product matrix of the axis vector
    axisCross = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )

    # Compute the rotation matrix
    rotMatrix = np.eye(3) * np.cos(theta) + axisCross * np.sin(theta) + np.outer(axis, axis) * (1 - np.cos(theta))

    # Apply the rotation matrix to the control points
    if inplace:
        points = geo.ctrlPnts.reshape(-1, geo.ctrlPnts.shape[-1])
        rotPoints = np.dot(points, rotMatrix)
        geo.ctrlPnts = rotPoints.reshape(geo.ctrlPnts.shape)
        return geo
    else:
        geoRotated = deepcopy(geo)
        points = geoRotated.ctrlPnts.reshape(-1, geoRotated.ctrlPnts.shape[-1])
        rotPoints = np.dot(points, rotMatrix)
        geoRotated.ctrlPnts = rotPoints.reshape(geoRotated.ctrlPnts.shape)
        return geoRotated


def translate(geo: GEOTYPE, vec: np.ndarray, inplace: bool = False) -> GEOTYPE:
    if inplace:
        geo.ctrlPnts = geo.ctrlPnts + vec
        return geo
    else:
        geoTrans = deepcopy(geo)
        geoTrans.ctrlPnts = geoTrans.ctrlPnts + vec
        return geoTrans
