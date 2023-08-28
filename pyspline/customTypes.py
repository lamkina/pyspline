# Standard Python modules
from typing import Union

# Local modules
from pyspline.bspline import BSplineCurve, BSplineSurface, BSplineVolume
from pyspline.nurbs import NURBSCurve, NURBSSurface

# --- Define custom aggregated types ---
GEOTYPE = Union[BSplineCurve, BSplineSurface, BSplineVolume, NURBSCurve, NURBSSurface]
CURVETYPE = Union[BSplineCurve, NURBSCurve]
SURFTYPE = Union[BSplineSurface, NURBSSurface]
NURBSTYPE = Union[NURBSCurve, NURBSSurface]
BSPLINETYPE = Union[BSplineCurve, BSplineSurface, BSplineVolume]
