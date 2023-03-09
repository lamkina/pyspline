# Standard Python modules
from typing import Optional, Tuple

# External modules
import numpy as np


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
