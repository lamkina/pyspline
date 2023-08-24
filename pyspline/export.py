# Standard Python modules
from typing import List, Optional, TextIO, Tuple

# External modules
import numpy as np

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .customTypes import CURVETYPE, GEOTYPE, SURFTYPE
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


def _writeIGESDirectoryCurve(curve: CURVETYPE, handle: TextIO, Dcount: int, Pcount: int, twoD=False) -> Tuple[int, int]:
    """
    Write the IGES file header information (Directory Entry Section)
    for this curve.

    DO NOT MODIFY ANYTHING HERE UNLESS YOU KNOW **EXACTLY** WHAT
    YOU ARE DOING!

    """

    if curve.nDim != 3:
        raise ValueError("Must have 3 dimensions to write to IGES file")

    paraEntries = 6 + len(curve.knotVec) + curve.nCtl + 3 * curve.nCtl + 5

    paraLines = (paraEntries - 11) // 3 + 2
    if np.mod(paraEntries - 11, 3) != 0:
        paraLines += 1
    if twoD:
        handle.write("     126%8d       0       0       1       0       0       001010501D%7d\n" % (Pcount, Dcount))
        handle.write(
            "     126       0       2%8d       0                               0D%7d\n" % (paraLines, Dcount + 1)
        )
    else:
        handle.write("     126%8d       0       0       1       0       0       000000001D%7d\n" % (Pcount, Dcount))
        handle.write(
            "     126       0       2%8d       0                               0D%7d\n" % (paraLines, Dcount + 1)
        )

    Dcount += 2
    Pcount += paraLines

    return Pcount, Dcount


def _writeIGESParameters(curve: CURVETYPE, handle: TextIO, Pcount: int, counter: int):
    """Write the IGES parameter information for this curve.

    DO NOT MODIFY ANYTHING HERE UNLESS YOU KNOW **EXACTLY** WHAT
    YOU ARE DOING!
    """
    handle.write(
        "%10d,%10d,%10d,0,0,0,0,                        %7dP%7d\n"
        % (126, curve.nCtl - 1, curve.degree, Pcount, counter)
    )
    counter += 1
    pos_counter = 0

    for i in range(len(curve.knotVec)):
        pos_counter += 1
        handle.write("%20.12g," % (np.real(curve.knotVec[i])))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0

    for _i in range(curve.nCtl):
        pos_counter += 1
        handle.write("%20.12g," % (1.0))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0

    for i in range(curve.nCtl):
        for idim in range(3):
            pos_counter += 1
            handle.write("%20.12g," % (np.real(curve.ctrlPnts[i, idim])))
            if np.mod(pos_counter, 3) == 0:
                handle.write("  %7dP%7d\n" % (Pcount, counter))
                counter += 1
                pos_counter = 0
    if pos_counter == 1:
        handle.write("%s    %7dP%7d\n" % (" " * 40, Pcount, counter))
        counter += 1
    elif pos_counter == 2:
        handle.write("%s    %7dP%7d\n" % (" " * 20, Pcount, counter))
        counter += 1

    # Ouput the ranges
    handle.write("%20.12g,%20.12g,0.0,0.0,0.0;         " % (np.min(curve.knotVec), np.max(curve.knotVec)))
    handle.write("  %7dP%7d\n" % (Pcount, counter))
    counter += 1
    Pcount += 2

    return Pcount, counter


def _writeIGESDirectorySurf(surf: SURFTYPE, handle: TextIO, Dcount: int, Pcount: int):
    """
    Write the IGES file header information (Directory Entry Section)
    for this surface
    """
    # A simpler calc based on cmlib definitions The 13 is for the
    # 9 parameters at the start, and 4 at the end. See the IGES
    # 5.3 Manual paraEntries = 13 + Knotsu + Knotsv + Weights +
    # control points
    if surf.nDim != 3:
        raise ValueError("Must have 3 dimensions to write to IGES file")
    paraEntries = (
        13 + (len(surf.uKnotVec)) + (len(surf.vKnotVec)) + surf.nCtlu * surf.nCtlv + 3 * surf.nCtlu * surf.nCtlv + 1
    )

    paraLines = (paraEntries - 10) // 3 + 2
    if np.mod(paraEntries - 10, 3) != 0:
        paraLines += 1

    # handle.write("1H,,1H$,1H.,1H,,1H$,16HSTANDARD,1.0,1.0,2HM,0.001,0,0,2HMM,1.0,0\n")
    handle.write("     128%8d       0       0       1       0       0       000000001D%7d\n" % (Pcount, Dcount))
    handle.write("     128       0       2%8d       0                               0D%7d\n" % (paraLines, Dcount + 1))
    Dcount += 2
    Pcount += paraLines

    return Pcount, Dcount


def _writeIGESParametersSurf(surf: SURFTYPE, handle: TextIO, Pcount: int, counter: int):
    """
    Write the IGES parameter information for this surface
    """
    handle.write(
        "%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n"
        % (128, surf.nCtlu - 1, surf.nCtlv - 1, surf.uDegree, surf.vDegree, Pcount, counter)
    )
    counter += 1
    handle.write("%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n" % (0, 0, 1, 0, 0, Pcount, counter))

    counter += 1
    pos_counter = 0

    for i in range(len(surf.uKnotVec)):
        pos_counter += 1
        handle.write("%20.12g," % (np.real(surf.uKnotVec[i])))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0
        # end if
    # end for

    for i in range(len(surf.vKnotVec)):
        pos_counter += 1
        handle.write("%20.12g," % (np.real(surf.vKnotVec[i])))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0
        # end if
    # end for

    for _i in range(surf.nCtlu * surf.nCtlv):
        pos_counter += 1
        handle.write("%20.12g," % (1.0))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0
        # end if
    # end for

    for j in range(surf.nCtlv):
        for i in range(surf.nCtlu):
            for idim in range(3):
                pos_counter += 1
                handle.write("%20.12g," % (np.real(surf.ctrlPnts[i, j, idim])))
                if np.mod(pos_counter, 3) == 0:
                    handle.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0
                # end if
            # end for
        # end for
    # end for

    # Ouput the ranges
    for i in range(4):
        pos_counter += 1
        if i == 0:
            handle.write("%20.12g," % (np.real(np.min(surf.uKnotVec))))
        if i == 1:
            handle.write("%20.12g," % (np.real(np.max(surf.uKnotVec))))
        if i == 2:
            handle.write("%20.12g," % (np.real(np.min(surf.vKnotVec))))
        if i == 3:
            # semi-colon for the last entity
            handle.write("%20.12g;" % (np.real(np.max(surf.vKnotVec))))
        if np.mod(pos_counter, 3) == 0:
            handle.write("  %7dP%7d\n" % (Pcount, counter))
            counter += 1
            pos_counter = 0
        else:  # We have to close it up anyway
            if i == 3:
                for _j in range(3 - pos_counter):
                    handle.write("%21s" % (" "))
                # end for
                pos_counter = 0
                handle.write("  %7dP%7d\n" % (Pcount, counter))
                counter += 1
            # end if
        # end if
    # end for

    Pcount += 2

    return Pcount, counter


def _writeCurveIGES(handle: TextIO, curve: CURVETYPE) -> None:
    Pcount, Dcount = _writeIGESDirectoryCurve(curve, handle, 1, 1)
    Pcount, counter = _writeIGESParameters(curve, handle, Pcount, 1)


def _writeSurfaceIGES(handle: TextIO, surf: SURFTYPE) -> None:
    Pcount, Dcount = _writeIGESDirectorySurf(surf, handle, 1, 1)
    Pcount, counter = _writeIGESParametersSurf(surf, handle, Pcount, 1)


def writeIGES(fileName: str, geo: GEOTYPE):
    with open(fileName, "w") as file:
        if isinstance(geo, BSplineCurve):
            _writeCurveIGES(file, geo)

        elif isinstance(geo, BSplineSurface):
            _writeSurfaceIGES(file, geo)
