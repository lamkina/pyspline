# Standard Python modules
from datetime import date
import time
from typing import List, Optional, TextIO, Tuple, Union

# External modules
import numpy as np

# First party modules
from pyspline import __version__

# Local modules
from .bspline import BSplineCurve, BSplineSurface
from .customTypes import CURVETYPE, GEOTYPE, SURFTYPE
from .operations import computeSurfaceNormals


class IGESWriter:
    def __init__(self, fileName: str, geoList: List[Union[CURVETYPE, SURFTYPE]], units: str = "m"):
        self.geoList = geoList
        if not self._validateGeometry():
            raise ValueError("One or more of the geometry inputs is not a valid geometry type.")

        self.geoInfo = self._classifyGeometry()

        self.m1 = 73  # First right margin
        self.m2 = 80  # Second right margin

        # Validate the units
        unitMap = {
            "inch": 1,
            "mm": 2,
            "ft": 4,
            "mi": 5,
            "m": 6,
            "km": 7,
            "mils": 8,
            "micron": 9,
            "cm": 10,
            "microinch": 11,
        }
        if units.lower() in unitMap.keys():
            self.unitName = units.capitalize()
            self.unitIdx = unitMap[units.lower()]
        else:
            raise ValueError(
                f"Unit input ({units}) was not found in the set of allowed units: {[key for key in unitMap.keys()]}."
            )

        # Set the filename
        self._setFileName(fileName)

        # Store the date
        self.date = date.today().strftime("%Y-%m-%d")
        self.timestamp = f"{date.today().strftime('%Y%m%d')}.{time.strftime('%H%M%S')}"

        # Get the maximum coordinate value
        self.coordMax = np.max(
            [np.max(geo.ctrlPntsW) if geo.rational else np.max(geo.ctrlPnts) for geo in self.geoList]
        )

        # Store the parameter and record delimiters
        self.paramDelim = ","
        self.recordDelim = ";"

    def _classifyGeometry(self) -> List[Tuple[str]]:
        geoInfo = []
        for geo in self.geoList:
            if isinstance(geo, BSplineCurve):
                geoType = "NURBSCurve" if geo.rational else "BSplineCurve"
                geoCode = "126"
            elif isinstance(geo, BSplineSurface):
                geoType = "NURBSSurface" if geo.rational else "BSplineSurface"
                geoCode = "128"

            geoInfo.append((geoType, geoCode))

        return geoInfo

    def _setFileName(self, fileName) -> None:
        # Determine the filename
        if fileName.endswith(".iges"):
            self.fileName = fileName
        else:
            if "." in fileName:
                name, ext = fileName.split(".")
                if ext == "igs":
                    print("IGES file format requires '.iges' extension not '.igs'...")
                    print("File extension will be changed to '.iges'.")
                    self.fileName = f"{name}.iges"
                else:
                    raise ValueError(f"File extension '{ext}' is not valid for IGES files.")

            else:
                self.fileName = f"{fileName}.iges"

    def _validateGeometry(self) -> bool:
        result = [isinstance(geo, (BSplineCurve, BSplineSurface)) for geo in self.geoList]

        if all(result):
            return True

        return False

    def _writeFirstLine(self, handle: TextIO):
        handle.write(f"{'S':>73}{1:>7}\n")

    def _formatString(self, text: str):
        nChar = len(text)
        return f"{nChar}H{text}"

    def _writeGlobalSection(self, handle: TextIO):
        # Make a list of strings to write in the global section
        strToWrite = []

        # Append each item to the strings to write list
        # NOTE: The order in which these are added matters!!  ** Do not change the order **
        strToWrite.append("")  # Parameter delimiter, empty string means accept default (,) (#1)
        strToWrite.append("")  # Record delimiter, empty string means accept default (;) (#2)
        strToWrite.append(self._formatString("pySpline IGES writer"))  # Product ID (#3)
        strToWrite.append(self._formatString(self.fileName))  # File name (#4)
        strToWrite.append(self._formatString(f"pySpline version {__version__}"))  # System ID (#5)
        strToWrite.append(self._formatString("STANDARD"))  # Preprocessor version (#6)
        strToWrite.append("32")  # Number of binary bits for integer representation (#7)
        strToWrite.append("308")  # Single precision magnitude (#8)
        strToWrite.append("15")  # Single precision significance (#9)
        strToWrite.append("308")  # Double precision magnitude (#10)
        strToWrite.append("15")  # Dobule precision significance (#11)
        strToWrite.append("")  # Product identification for the receiver (#12)
        strToWrite.append(f"1.")  # Model space scale (#13)
        strToWrite.append(f"{self.unitIdx}")  # Unit flag (#14)
        strToWrite.append(self._formatString(self.unitName))  # Units (#15)
        strToWrite.append("1")  # Maximum number of line weight gradiations (#16)
        strToWrite.append("0.01")  # Size of maximum line width in units (#17)
        strToWrite.append(self._formatString(self.timestamp))  # Timestamp (#18)
        strToWrite.append("1E-07")  # Minimum user-intended resolution (#19)
        strToWrite.append(f"{self.coordMax:.5f}")  # Approximate maximum coordinate value (#20)
        strToWrite.append("")  # Name of author (#21)
        strToWrite.append("")  # Author's Organization (#22)
        strToWrite.append("11")  # IGES version number (#23)
        strToWrite.append("0")  # Drafting code standard (#24)

        # Loop over the values to format and write lines
        lineSum = 0
        lineCount = 1
        lineText = ""
        for i, string in enumerate(strToWrite):
            lineSum += len(string) + 1  # Add 1 for the delimiter

            if lineSum > (self.m1 - 1):
                # We will be over the limit, write the line
                handle.write(f"{lineText}{'G':>{self.m1 - len(lineText)}}{lineCount:>{self.m2 - self.m1}}\n")

                # Reset the lineSum
                lineText = string + ","  # Reset the line text to the previous string
                lineSum = len(lineText)  # Reset the line sum
                lineCount += 1  # Increment the line counter
            else:
                if i == len(strToWrite) - 1:
                    # We are at the last element of the list and didn't write yet, so just write what's left
                    lineText += (
                        string + self.paramDelim + self.recordDelim
                    )  # Replace the trailing comma with the record delimiter
                    handle.write(f"{lineText}{'G':>{self.m1 - len(lineText)}}{lineCount:>{self.m2 - self.m1}}\n")

                lineText += string + ","

    def _getStatusNumber(self, geo: Union[CURVETYPE, SURFTYPE]) -> str:
        # 00 means entity is to be displayed
        blankStatus = "00"

        # 00 means entity is independent
        entitySwitch = "00"

        # 00 means full 3D geometry
        entityUseFlag = "00"

        # 01 means no previous directory entry attributes will apply
        hierarchy = "01"

        return blankStatus + entitySwitch + entityUseFlag + hierarchy

    def _getParamCounts(self, geo: Union[CURVETYPE, SURFTYPE]) -> Tuple[int]:
        if isinstance(geo, BSplineCurve):
            paramEntries = len(geo.knotVec) + geo.nCtl + (3 * geo.nCtl) + 5
            paramLines = int(np.ceil(paramEntries / 3)) + 1

        elif isinstance(geo, BSplineSurface):
            paramEntries = (
                13 + (len(geo.uKnotVec) + len(geo.vKnotVec)) + geo.nCtlu * geo.nCtlv + 3 * geo.nCtlu * geo.nCtlv + 1
            )
            paramLines = (paramEntries - 10) // 3 + 1
            if np.mod(paramEntries - 10, 3) != 0:
                paramLines += 1
        return paramEntries, paramLines

    def _writeDirectoryEntry(self, handle: TextIO) -> Tuple[int]:
        pCount = 1
        lineCounter = 1
        for i, geo in enumerate(self.geoList):
            entityType = self.geoInfo[i][1]
            row1 = []
            row1.append(entityType)  # Entity type number (#1)
            row1.append(f"{pCount}")  # Parameter data pointer (#2)
            row1.append("0")  # Structure (#3)
            row1.append("0")  # Line font pattern, 1 is solid line (#4)
            row1.append("0")  # Entity level (#5)
            row1.append("0")  # Viewing options, 0 is equal visibility (#6)
            row1.append("0")  # Transformation matrix pointer, 0 is no transformations (#7)
            row1.append("0")  # Label display associativity, 0 is no associativity (#8)

            statusNumber = self._getStatusNumber(geo)
            row1.append(statusNumber)  # Status number (#9)

            row1Text = ""
            for string in row1:
                row1Text += f"{string:>8}"

            row1Text += "D"
            row1Text += f"{lineCounter:>7}\n"
            lineCounter += 1

            handle.write(row1Text)

            row2 = []
            row2.append(entityType)  # Entity type number (#1)
            row2.append("0")  # Line weight number (#2)
            row2.append("0")  # Line color number (#3)

            _, paramLines = self._getParamCounts(geo)
            pCount += paramLines
            row2.append(f"{paramLines}")  # Parameter line count number (#4)
            row2.append("0")  # Form number (#5)
            row2.append(" ")  # Reserved field **not used** (#6)
            row2.append(" ")  # Reserved field **not used** (#7)
            row2.append(" ")  # Entity label **not used** (#8)
            row2.append("0")  # Subscript number (#9)

            row2Text = ""
            for string in row2:
                row2Text += f"{string:>8}"

            row2Text += "D"
            row2Text += f"{lineCounter:>7}\n"
            lineCounter += 1

            handle.write(row2Text)

        return lineCounter, pCount

    def _writeParameterDataCurve(self, geo: CURVETYPE, handle: TextIO, pCount: int, counter: int):
        entryCode = 126
        k = geo.nCtl - 1
        m = geo.degree
        n = k - m + 1
        a = n + (2 * m)

        nDim = geo.nDim - 1 if geo.rational else geo.nDim  # Adjust for homegenous coords if rational
        prop1 = 0 if geo.nDim > 2 else 1  # nonplanar (always assumed)

        # Check the first and last control points to determine if the curve is open or closed
        if np.allclose(geo.ctrlPnts[0], geo.ctrlPnts[-1], atol=1e-14):
            prop2 = 1
        else:
            prop2 = 0

        prop3 = 0  # Curve will always be rational
        prop4 = 0  # Non-periodic (always assumed for now)

        paramText = f"{pCount}P"
        lineText = f"{entryCode:>8d},{k:>7d},{m:>7d},{prop1:>7d},{prop2:>7d},{prop3:>7d},{prop4:>7d},"
        lineText += f"{paramText:>{self.m1 - len(lineText)}}{counter:>{self.m2 - self.m1}}\n"
        handle.write(lineText)
        counter += 1

        strToWrite = []
        # Add the knot vector from index 7 to index 7+a (defined above)
        for knot in geo.knotVec:
            strToWrite.append(f"{knot:20.12G}")

        # Add the weights from index 8+a to 8+a+k (defined above)
        if geo.rational:
            weights = geo.weights
        else:
            weights = np.ones(geo.nCtl)  # Weights are array of ones of length nCtl

        for weight in weights:
            strToWrite.append(f"{weight:20.12G}")

        # Add the control points (x, y, z are written consecutively)
        # From index 9+a+k to 11+a+4*k (defined above)

        # Get the weighted control points if rational, else just get the regular control points
        # because the weights will all be equal to one
        ctrlPnts = geo.ctrlPntsW[:, :-1] if geo.rational else geo.ctrlPnts

        # We need to add a zero y-coordinate to planar curves
        if nDim < 3:
            ctrlPnts = np.column_stack((ctrlPnts, np.zeros(geo.nCtl)))

        for i in range(geo.nCtl):
            for iDim in range(3):
                strToWrite.append(f"{ctrlPnts[i, iDim]:20.12G}")

        # Get the start and end parameter values (we know these will always be 0 to 1)
        strToWrite.append(f"{0:20.12g}")
        strToWrite.append(f"{1:20.12g}")

        # Write the planar normal vector (we dont use these so leave them blank)
        if prop1 == 1:
            strToWrite.append(f"{0:20}")
            strToWrite.append(f"{0:20}")
            strToWrite.append(f"{1:20}")
        else:
            strToWrite.append(f"{' ':20}")
            strToWrite.append(f"{' ':20}")
            strToWrite.append(f"{' ':20}")

        # Now we need to write all the data
        # Loop over the values to format and write lines
        lineCount = counter
        currentLine = ""
        for i, string in enumerate(strToWrite):
            spaceNeeded = len(string) + 1  # add 1 for the delimiter

            if len(currentLine) + spaceNeeded < 64:
                if currentLine:
                    currentLine += ","

                currentLine += string

                if i == len(strToWrite) - 1:
                    currentLine += ";"
            else:
                currentLine += ","
                handle.write(
                    f"{currentLine}{paramText:>{self.m1 - len(currentLine)}}{lineCount:>{self.m2 - self.m1}}\n"
                )
                lineCount += 1
                currentLine = string

                if i == len(strToWrite) - 1:
                    currentLine += ";"

        if currentLine:
            handle.write(f"{currentLine}{paramText:>{self.m1 - len(currentLine)}}{lineCount:>{self.m2 - self.m1}}\n")

        lineCount += 1

        return lineCount

    def _writeParameterDataSurf(self, handle: TextIO):
        pass

    def _writeParameterData(self, handle: TextIO):
        pCount = 1
        counter = 1
        for geo in self.geoList:
            if isinstance(geo, BSplineCurve):
                counter = self._writeParameterDataCurve(geo, handle, pCount, counter)
            elif isinstance(geo, BSplineSurface):
                counter = self._writeParameterDataSurf(handle)

            pCount += 2

        return pCount, counter

    def _writeTerminate(self, handle: TextIO, dCount: int, pCount: int) -> None:
        sTag = f"S{1:>7}"
        gTag = f"G{3:>7}"
        dTag = f"D{dCount-1:>7}"
        pTag = f"P{pCount-1:>7}"
        lineText = f"{sTag}{gTag}{dTag}{pTag}"
        lineText += f"{'T':>{self.m1-len(lineText)}}{1:>{self.m2-self.m1}}"
        handle.write(lineText)

    def write(self):
        with open(self.fileName, "w") as file:
            self._writeFirstLine(file)
            self._writeGlobalSection(file)
            dCount, pCount = self._writeDirectoryEntry(file)
            pCount, counter = self._writeParameterData(file)
            self._writeTerminate(file, dCount, counter)


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


def writeIGES(
    fileName: str,
    geo: Union[List[Union[BSplineCurve, BSplineSurface]], Union[BSplineCurve, BSplineSurface]],
    units: str = "m",
):
    if not isinstance(geo, list) and isinstance(geo, (BSplineCurve, BSplineSurface)):
        geo = [geo]

    writer = IGESWriter(fileName, geo, units=units)
    writer.write()
