# External modules
import numpy as np

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .operations import computeCurveData, computeSurfaceData
from .utils import closeTecplot, openTecplot, writeTecplot1D, writeTecplot2D
from .custom_types import GEOTYPE


def writeSurfaceDirections(surf: BSplineSurface, file: str, isurf: int):
    if surf.nCtlu >= 3 and surf.nCtlv >= 3:
        data = np.zeros((4, surf.nDim))
        data[0] = surf.ctrlPnts[1, 2]
        data[1] = surf.ctrlPnts[1, 1]
        data[2] = surf.ctrlPnts[2, 1]
        data[3] = surf.ctrlPnts[3, 1]
        writeTecplot1D(file, f"surface{isurf}_direction", data)
    else:
        print("Not enough control points to output direction indicator")


def writeTecplot(geo: GEOTYPE, fileName: str, **kwargs):
    file = openTecplot(fileName, geo.nDim)

    # Curve keyword arguments
    curve = kwargs.get("curve", True)

    # Surface keyword arguments
    surf = kwargs.get("surf", True)
    directions = kwargs.get("directions", False)

    # Shared keyword arguments
    control_points = kwargs.get("control_points", True)
    orig = kwargs.get("orig", True)

    # Tecplot keyword args
    solutionTime = kwargs.get("solutionTime", None)

    if isinstance(geo, BSplineCurve):
        if curve:
            data = computeCurveData(geo)
            writeTecplot1D(file, "interpolated", data, solutionTime=solutionTime)
        if control_points:
            writeTecplot1D(file, "control_points", geo.ctrlPnts, solutionTime=solutionTime)
            if geo._rational:
                writeTecplot1D(file, "weighted_cpts", geo.ctrlPntsW[:, :3], solutionTime=solutionTime)
        if orig and geo.X is not None:
            writeTecplot1D(file, "orig_data", geo.X, solutionTime=solutionTime)
    elif isinstance(geo, BSplineSurface):
        if surf:
            data = computeSurfaceData(geo)
            writeTecplot2D(file, "interpolated", data)
        if control_points:
            writeTecplot2D(file, "control_points", geo.ctrlPnts)
        if directions:
            writeSurfaceDirections(surf, file, 0)
    else:
        pass

    closeTecplot(file)
