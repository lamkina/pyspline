# Standard Python modules
from typing import List, Optional, TextIO

# External modules
import numpy as np

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .customTypes import GEOTYPE, SURFTYPE
from .operations import computeSurfaceNormals


def writeTecplot1D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solution_time: Optional[int] = None,
):
    """A Generic function to write a 1D data zone to a tecplot file.

    Parameters
    ----------
    handle : file handle
        Open file handle
    name : str
        Name of the zone to use
    data : array of size (N, ndim)
        1D array of data to write to file
    solution_time : float
        Solution time to write to the file. This could be a fictitious time to
        make visualization easier in tecplot.
    """
    ni = data.shape[0]
    ndim = data.shape[1]

    if len(variables) != ndim:
        raise ValueError("The length of the variables must equal the dimensionality of the data.")

    _writeTecplotHeader(handle, name, solution_time, variables, ni)

    for i in range(ni):
        for idim in range(ndim):
            handle.write(f"{data[i, idim]} ")
        handle.write("\n")


def writeTecplot2D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solution_time=None,
):
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
    ni = data.shape[0]
    nj = data.shape[1]
    ndim = data.shape[2]

    if len(variables) != ndim:
        raise ValueError("The length of the variables must equal the dimensionality of the data.")

    _writeTecplotHeader(handle, name, solution_time, variables, ni, nj)

    for j in range(nj):
        for i in range(ni):
            for idim in range(ndim):
                handle.write(f"{data[i, j, idim]} " % (data[i, j, idim]))
            handle.write("\n")


def writeTecplot3D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solution_time=None,
):
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
    ni = data.shape[0]
    nj = data.shape[1]
    nk = data.shape[2]
    ndim = data.shape[3]

    if len(variables) != ndim:
        raise ValueError("The length of the variables must equal the dimensionality of the data.")

    _writeTecplotHeader(handle, name, solution_time, variables, ni, nj, nk)

    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                for idim in range(ndim):
                    handle.write(f"{data[i, j, k, idim]} ")
                handle.write("\n")


def writeTecplotNormals(fileName, name, data, solutionTime=None):
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
    handle = open(fileName, "w")
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]
    ndim = data.shape[3]

    handle.write('VARIABLES = "CoordinateX", "CoordinateY", "CoordinateZ", "NormX", "NormY", "NormZ"\n')
    handle.write(f'Zone T="{name}" I={nx} J={ny} K={nz}\n')
    if solutionTime is not None:
        handle.write(f"SOLUTIONTIME={solutionTime}\n")
    handle.write("DATAPACKING=POINT\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for idim in range(ndim):
                    handle.write("%20.16g " % (data[i, j, k, idim]))
                handle.write("\n")


def _writeTecplotHeader(
    handle: TextIO,
    zone_name: str,
    solution_time: int,
    variables: List[str],
    ni: Optional[int] = None,
    nj: Optional[int] = None,
    nk: Optional[int] = None,
):
    """Write tecplot variable header"""
    var_str = ", ".join([f'"{var}"' for var in variables])
    handle.write(f"VARIABLES = {var_str}\n")

    zone_dim_str = ""
    if ni is not None:
        zone_dim_str += f"I={ni} "
    if nj is not None:
        zone_dim_str += f"J={nj} "
    if nk is not None:
        zone_dim_str += f"K={nk} "

    handle.write(f'Zone T="{zone_name}" {zone_dim_str}\n')

    if solution_time is not None:
        handle.write(f"SOLUTIONTIME={solution_time}\n")

    handle.write("DATAPACKING=POINT\n")


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


def writeSurfaceNormalsTecplot(surf: SURFTYPE, fileName: str) -> None:
    surf.computeData(recompute=True)
    norm_vecs = computeSurfaceNormals(surf.uData, surf.vData, surf)
    data = np.concatenate((surf.data, norm_vecs), axis=2).reshape((len(surf.uData), len(surf.vData), 1, 6))
    variables = ["CoordinateX", "CoordinateY", "CoordinateZ", "NormX", "NormY", "NormZ"]

    with open("surface_normals.dat", "w") as file:
        writeTecplot3D(file, "normals", data, variables)


def writeTecplot(geo: GEOTYPE, fileName: str, **kwargs):
    # Curve keyword arguments
    curve = kwargs.get("curve", True)

    # Surface keyword arguments
    surf = kwargs.get("surf", True)
    directions = kwargs.get("directions", False)

    normals = kwargs.get("normals", False)

    # Shared keyword arguments
    control_points = kwargs.get("control_points", True)
    orig = kwargs.get("orig", True)

    # Tecplot keyword args
    solutionTime = kwargs.get("solutionTime", None)

    # Compute the postprocessing data (all geo types share this method)
    geo.computeData(recompute=True)

    with open(fileName, "w") as file:
        if isinstance(geo, BSplineCurve):
            if curve:
                writeTecplot1D(file, "interpolated", geo.data[:, :3], solutionTime=solutionTime)
            if control_points:
                writeTecplot1D(file, "control_points", geo.ctrlPnts, solutionTime=solutionTime)
                if geo.rational:
                    writeTecplot1D(file, "weighted_cpts", geo.ctrlPntsW[:, :-1], solutionTime=solutionTime)
            if orig and geo.X is not None:
                writeTecplot1D(file, "orig_data", geo.X, solutionTime=solutionTime)
        elif isinstance(geo, BSplineSurface):
            if surf:
                writeTecplot2D(file, "interpolated", geo.data, solutionTime=solutionTime)
            if control_points:
                writeTecplot2D(file, "control_points", geo.ctrlPnts, solutionTime=solutionTime)
                if geo.rational:
                    writeTecplot2D(file, "weighted_cpts", geo.ctrlPntsW[:, :, :-1], solutionTime=solutionTime)
            if orig and geo.X is not None:
                writeTecplot2D(file, "orig_data", geo.X, solutionTime=solutionTime)
            if directions:
                writeSurfaceDirections(surf, file, 0)
        else:
            pass
