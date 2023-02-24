# Standard Python modules
from typing import Union

# Local modules
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume

# --- Define custom aggregated types ---
GEOTYPE = Union[BSplineCurve, BSplineSurface, BSplineVolume]
