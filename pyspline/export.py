# External modules
import numpy as np

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .customTypes import GEOTYPE
from .operations import computeCurveData, computeSurfaceData


def writeTecplot1D(handle, name, data, solutionTime=None):
    """A Generic function to write a 1D data zone to a tecplot file.
    Parameters
    ----------
    handle : file handle
        Open file handle
    name : str
        Name of the zone to use
    data : array of size (N, ndim)
        1D array of data to write to file
    SolutionTime : float
        Solution time to write to the file. This could be a fictitious time to
        make visualization easier in tecplot.
    """
    nx = data.shape[0]
    ndim = data.shape[1]
    handle.write('Zone T="%s" I=%d\n' % (name, nx))
    if solutionTime is not None:
        handle.write("SOLUTIONTIME=%f\n" % (solutionTime))
    handle.write("DATAPACKING=POINT\n")
    for i in range(nx):
        for idim in range(ndim):
            handle.write("%f " % (data[i, idim]))
        handle.write("\n")


def writeTecplot2D(handle, name, data, solutionTime=None):
    """A Generic function to write a 2D data zone to a tecplot file.
    Parameters
    ----------
    handle : file handle
        Open file handle
    name : str
        Name of the zone to use
    data : 2D np array of size (nx, ny, ndim)
        2D array of data to write to file
    SolutionTime : float
        Solution time to write to the file. This could be a fictitious time to
        make visualization easier in tecplot.
    """
    nx = data.shape[0]
    ny = data.shape[1]
    ndim = data.shape[2]
    handle.write('Zone T="%s" I=%d J=%d\n' % (name, nx, ny))
    if solutionTime is not None:
        handle.write("SOLUTIONTIME=%f\n" % (solutionTime))
    handle.write("DATAPACKING=POINT\n")
    for j in range(ny):
        for i in range(nx):
            for idim in range(ndim):
                handle.write("%20.16g " % (data[i, j, idim]))
            handle.write("\n")


def writeTecplot3D(handle, name, data, solutionTime=None):
    """A Generic function to write a 3D data zone to a tecplot file.
    Parameters
    ----------
    handle : file handle
        Open file handle
    name : str
        Name of the zone to use
    data : 3D np array of size (nx, ny, nz, ndim)
        3D array of data to write to file
    SolutionTime : float
        Solution time to write to the file. This could be a fictitious time to
        make visualization easier in tecplot.
    """
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]
    ndim = data.shape[3]
    handle.write('Zone T="%s" I=%d J=%d K=%d\n' % (name, nx, ny, nz))
    if solutionTime is not None:
        handle.write("SOLUTIONTIME=%f\n" % (solutionTime))
    handle.write("DATAPACKING=POINT\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for idim in range(ndim):
                    handle.write("%f " % (data[i, j, k, idim]))
                handle.write("\n")


def _writeHeader(f, ndim):
    """Write tecplot zone header depending on spatial dimension"""
    if ndim == 1:
        f.write('VARIABLES = "CoordinateX"\n')
    elif ndim == 2:
        f.write('VARIABLES = "CoordinateX", "CoordinateY"\n')
    else:
        f.write('VARIABLES = "CoordinateX", "CoordinateY", "CoordinateZ"\n')


def openTecplot(fileName, ndim):
    """A Generic function to open a Tecplot file to write spatial data.

    Parameters
    ----------
    fileName : str
        Tecplot filename. Should have a .dat extension.
    ndim : int
        Number of spatial dimensions. Must be 1, 2 or 3.

    Returns
    -------
    f : file handle
        Open file handle
    """
    f = open(fileName, "w")
    _writeHeader(f, ndim)

    return f


def closeTecplot(f):
    """Close Tecplot file opened with openTecplot()"""
    f.close()


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
            if geo.rational:
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
