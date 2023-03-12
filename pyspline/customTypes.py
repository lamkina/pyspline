# Standard Python modules
from typing import Union

# Local modules
from .bspline import BSplineCurve, BSplineSurface, BSplineVolume
from .nurbs import NURBSCurve, NURBSSurface

# --- Define custom aggregated types ---
GEOTYPE = Union[BSplineCurve, BSplineSurface, BSplineVolume, NURBSCurve, NURBSSurface]
CURVETYPE = Union[BSplineCurve, NURBSCurve]
SURFTYPE = Union[BSplineSurface, NURBSSurface]
