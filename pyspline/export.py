# Standard Python modules
from pathlib import Path, PosixPath
from typing import List, Optional, TextIO, Union

# External modules
import numpy as np

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .customTypes import GEOTYPE, SURFTYPE
from .iges.igesWriter import IGESWriter
from .operations import computeSurfaceNormals


def writeTecplot1D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solutionTime: Optional[int] = None,
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
    solutionTime : float
        Solution time to write to the file. This could be a fictitious time to
        make visualization easier in tecplot.
    """
    ni = data.shape[0]
    ndim = data.shape[1]

    if len(variables) != ndim:
        raise ValueError("The length of the variables must equal the dimensionality of the data.")

    _writeTecplotHeader(handle, name, solutionTime, variables, ni)

    for i in range(ni):
        for idim in range(ndim):
            handle.write(f"{data[i, idim]} ")
        handle.write("\n")


def writeTecplot2D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solutionTime=None,
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

    _writeTecplotHeader(handle, name, solutionTime, variables, ni, nj)

    for j in range(nj):
        for i in range(ni):
            for idim in range(ndim):
                handle.write(f"{data[i, j, idim]} ")
            handle.write("\n")


def writeTecplot3D(
    handle: TextIO,
    name: str,
    data: np.ndarray,
    variables: List[str] = ["CoordinateX", "CoordinateY", "CoordinateZ"],
    solutionTime=None,
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

    _writeTecplotHeader(handle, name, solutionTime, variables, ni, nj, nk)

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
    zoneName: str,
    solutionTime: int,
    variables: List[str],
    ni: Optional[int] = None,
    nj: Optional[int] = None,
    nk: Optional[int] = None,
):
    """Write tecplot variable header"""
    varStr = ", ".join([f'"{var}"' for var in variables])
    handle.write(f"VARIABLES = {varStr}\n")

    zoneDimStr = ""
    if ni is not None:
        zoneDimStr += f"I={ni} "
    if nj is not None:
        zoneDimStr += f"J={nj} "
    if nk is not None:
        zoneDimStr += f"K={nk} "

    handle.write(f'Zone T="{zoneName}" {zoneDimStr}\n')

    if solutionTime is not None:
        handle.write(f"SOLUTIONTIME={solutionTime}\n")

    handle.write("DATAPACKING=POINT\n")


def writeSurfaceDirections(surf: BSplineSurface, file: str, isurf: int):
    if surf.nCtlu >= 3 and surf.nCtlv >= 3:
        data = np.zeros((4, surf.nDim))
        data[0] = surf.ctrlPnts[1, 2]
        data[1] = surf.ctrlPnts[1, 1]
        data[2] = surf.ctrlPnts[2, 1]
        data[3] = surf.ctrlPnts[3, 1]
        writeTecplot1D(file, f"surface{isurf}Direction", data)
    else:
        print("Not enough control points to output direction indicator")


def writeSurfaceNormalsTecplot(surf: SURFTYPE, fileName: str) -> None:
    surf.computeData(recompute=True)
    normVecs = computeSurfaceNormals(surf.uData, surf.vData, surf)
    data = np.concatenate((surf.data, normVecs), axis=2).reshape((len(surf.uData), len(surf.vData), 1, 6))
    variables = ["CoordinateX", "CoordinateY", "CoordinateZ", "NormX", "NormY", "NormZ"]

    with open(fileName, "w") as file:
        writeTecplot3D(file, "normals", data, variables)


def writeTecplot(geo: GEOTYPE, fileName: str, **kwargs):
    # Curve keyword arguments
    curve = kwargs.get("curve", True)

    # Surface keyword arguments
    surf = kwargs.get("surf", True)
    directions = kwargs.get("directions", False)

    # Shared keyword arguments
    controlPoints = kwargs.get("controlPoints", True)
    orig = kwargs.get("orig", True)

    # Tecplot keyword args
    solutionTime = kwargs.get("solutionTime", None)

    # Compute the postprocessing data (all geo types share this method)
    geo.computeData(recompute=True)

    if geo.nDim == 1:
        variables = ["CoordinateX"]
    elif geo.nDim == 2:
        variables = ["CoordinateX", "CoordinateY"]
    elif geo.nDim == 3:
        variables = ["CoordinateX", "CoordinateY", "CoordinateZ"]

    with open(fileName, "w") as file:
        if isinstance(geo, BSplineCurve):
            if curve:
                writeTecplot1D(file, "interpolated", geo.data, variables=variables, solutionTime=solutionTime)
            if controlPoints:
                writeTecplot1D(file, "controlPoints", geo.ctrlPnts, variables=variables, solutionTime=solutionTime)
                if geo.rational:
                    writeTecplot1D(
                        file, "weightedCpts", geo.ctrlPntsW[:, :-1], variables=variables, solutionTime=solutionTime
                    )
            if orig and geo.X is not None:
                writeTecplot1D(file, "origData", geo.X, variables=variables, solutionTime=solutionTime)
        elif isinstance(geo, BSplineSurface):
            if surf:
                writeTecplot2D(file, "interpolated", geo.data, variables=variables, solutionTime=solutionTime)
            if controlPoints:
                writeTecplot2D(file, "controlPoints", geo.ctrlPnts, variables=variables, solutionTime=solutionTime)
                if geo.rational:
                    writeTecplot2D(
                        file, "weightedCpts", geo.ctrlPntsW[:, :, :-1], variables=variables, solutionTime=solutionTime
                    )
            if orig and geo.X is not None:
                writeTecplot2D(file, "origData", geo.X, variables=variables, solutionTime=solutionTime)
            if directions:
                writeSurfaceDirections(surf, file, 0)
        else:
            pass


def writeIGES(fileName: Union[str, PosixPath], geo: Union[List[GEOTYPE], GEOTYPE], units: str = "m", **kwargs):
    if not isinstance(geo, list):
        geo = [geo]

    coordMax, geoList = IGESWriter.preprocess(geo)

    fileName = Path(fileName) if not isinstance(fileName, PosixPath) else fileName
    f = IGESWriter.openFile(fileName)
    IGESWriter.writeFirstLine(f)
    gCount = IGESWriter.writeGlobalSection(f, units, coordMax, **kwargs)

    pCount = 1
    dCount = 1
    for geo in geoList:
        paramLines, _ = IGESWriter.getParameterInfo(geo)
        if isinstance(geo, BSplineCurve):
            dCount = IGESWriter.writeDirectoryCurve(f, paramLines, pCount, dCount)
        elif isinstance(geo, BSplineSurface):
            dCount = IGESWriter.writeDirectorySurface(f, paramLines, pCount, dCount)
        pCount += paramLines

    pCount = 1
    counter = 1
    for geo in geoList:
        if isinstance(geo, BSplineCurve):
            counter = IGESWriter.writeParametersCurve(f, geo, pCount, counter)
        elif isinstance(geo, BSplineSurface):
            counter = IGESWriter.writeParametersSurface(f, geo, pCount, counter)
        pCount += 2

    IGESWriter.writeTerminateEntry(f, gCount, dCount, counter)
    f.close()
