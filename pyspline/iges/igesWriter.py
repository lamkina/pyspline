# Standard Python modules
from datetime import date
import time
from typing import List, TextIO, Tuple, Union

# External modules
import numpy as np

# Local modules
from .. import __version__
from ..bspline import BSplineCurve, BSplineSurface
from ..compatibility import combineCtrlPnts
from ..customTypes import GEOTYPE
from ..nurbs import NURBSCurve, NURBSSurface

# ==============================================================================
# IGES FORMATTING CONSTANTS
# ==============================================================================
# Map for the units
UNITMAP = {
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

# Delimiters
PARAMDELIM = ","
RECORDDELIM = ";"

# Right margins
MARGIN1 = 73
MARGIN2 = 80

# Parameter data formatting
PARAMMARGIN = 64
PARAMCOLS = 3
PARAMCOLWIDTH = 20
PARAMPRECISION = 12


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def formatString(string: str) -> str:
    return f"{len(string)}H{string}"


def formatParameterValue(val: float) -> None:
    return f"{val:{PARAMCOLWIDTH}.{PARAMPRECISION}G}"


def writeLines(handle: TextIO, lines: List[str], counter: int, marginText: str, margin: int) -> int:
    lineCount = counter
    currentLine = ""
    for i, string in enumerate(lines):
        spaceNeeded = len(string) + 1  # add 1 for the delimiter

        if len(currentLine) + spaceNeeded < margin:
            if currentLine:
                currentLine += ","

            currentLine += string

            if i == len(lines) - 1:
                currentLine += ";"
        else:
            currentLine += ","
            handle.write(f"{currentLine}{marginText:>{MARGIN1 - len(currentLine)}}{lineCount:>{MARGIN2 - MARGIN1}}\n")
            lineCount += 1
            currentLine = string

            if i == len(lines) - 1:
                currentLine += ";"

    if currentLine:
        handle.write(f"{currentLine}{marginText:>{MARGIN1 - len(currentLine)}}{lineCount:>{MARGIN2 - MARGIN1}}\n")

    lineCount += 1

    return lineCount


# ==============================================================================
# UTILITY CLASS WITH STATIC HELPER METHODS TO WRITE IGES FILES
# ==============================================================================
class IGESWriter:
    @staticmethod
    def openFile(fileName: str) -> TextIO:
        # Determine the filename
        if fileName.endswith(".iges") or fileName.endswith(".igs"):
            fileName = fileName
        else:
            if "." in fileName:
                _, ext = fileName.split(".")
                raise ValueError(f"File extension '{ext}' is not valid for IGES files.")
            else:
                fileName = f"{fileName}.iges"

        return open(fileName, "w")

    @staticmethod
    def preprocess(geoList: List[GEOTYPE]) -> Tuple[float, List[Union[NURBSCurve, NURBSSurface]]]:
        coordMax = 0.0
        for i, geo in enumerate(geoList):
            if not isinstance(geo, (BSplineCurve, BSplineSurface)):
                raise ValueError("IGES can only write BSpline or NURBS surfaces and curves.")

            if not geo.rational:
                ctrlPntsW = combineCtrlPnts(geo.ctrlPnts)

            if isinstance(geo, BSplineCurve):
                newGeo = NURBSCurve(geo.degree, geo.knotVec, ctrlPntsW)
            elif isinstance(geo, NURBSSurface):
                newGeo = NURBSSurface(geo.uDegree, geo.vDegree, ctrlPntsW, geo.uKnotVec, geo.vKnotVec)

            geoList[i] = newGeo

            coordMax = max(coordMax, float(np.max(np.abs(newGeo.ctrlPntsW))))

        return coordMax, geoList

    @staticmethod
    def getParameterInfo(geo: Union[NURBSCurve, NURBSSurface]) -> Tuple[int, int]:
        if isinstance(geo, BSplineCurve):
            paramEntries = len(geo.knotVec) + len(geo.weights) + (3 * geo.nCtl) + 5
            paramLines = int(np.ceil(paramEntries / PARAMCOLS)) + 1

        elif isinstance(geo, BSplineSurface):
            paramEntries = (
                (len(geo.uKnotVec) + len(geo.vKnotVec)) + (geo.nCtlu * geo.nCtlv) + (3 * geo.nCtlu * geo.nCtlv) + 9
            )
            paramLines = int(np.ceil(paramEntries / PARAMCOLS)) + 1

        return paramLines, paramEntries

    @staticmethod
    def writeTerminateEntry(handle: TextIO, gCount: int, dCount: int, pCount: int) -> None:
        sTag = f"S{1:>7}"
        gTag = f"G{gCount-1:>7}"
        dTag = f"D{dCount-1:>7}"
        pTag = f"P{pCount-1:>7}"
        lineText = f"{sTag}{gTag}{dTag}{pTag}"
        lineText += f"{'T':>{MARGIN1-len(lineText)}}{1:>{MARGIN2-MARGIN1}}"
        handle.write(lineText)

    @staticmethod
    def writeGlobalSection(handle: TextIO, unit: str, coordMax: float, **kwargs) -> int:
        productID = kwargs.get("productID", "pySpline IGES Writer")
        minResolution = kwargs.get("minResolution", 1e-7)
        author = kwargs.get("author", "")
        authorOrg = kwargs.get("authorOrg", "")

        # Figure out the units
        if unit.lower() in UNITMAP.keys():
            unitName = unit.capitalize()
            unitIdx = UNITMAP[unit.lower()]
        else:
            raise ValueError(
                f"Unit input ({unit}) was not found in the set of allowed units: {[key for key in UNITMAP.keys()]}."
            )

        # Get the timestamp
        timeStamp = f"{date.today().strftime('%Y%m%d')}.{time.strftime('%H%M%S')}"

        # Make a list of strings to write in the global section
        # !!! Attention !!! The order in which these are added matters.  ** Do not change the order **
        lines = [
            formatString(PARAMDELIM),  # Parameter delimiter, empty string means accept default (,) (#1)
            formatString(RECORDDELIM),  # Record delimiter, empty string means accept default (;) (#2)
            formatString(productID),  # Product ID (#3)
            formatString(handle.name),  # File name (#4)
            formatString(f"pySpline version {__version__}"),  # System ID (#5)
            formatString("STANDARD"),  # Preprocessor version (#6)
            "32",  # Number of binary bits for integer representation (#7)
            "308",  # Single precision magnitude (#8)
            "15",  # Single precision significance (#9)
            "308",  # Double precision magnitude (#10)
            "15",  # Dobule precision significance (#11)
            "",  # Product identification for the receiver (#12)
            f"1.",  # Model space scale (#13)
            f"{unitIdx}",  # Unit flag (#14)
            formatString(unitName),  # Units (#15)
            "1",  # Maximum number of line weight gradiations (#16)
            "0.01",  # Size of maximum line width in units (#17)
            formatString(timeStamp),  # Timestamp (#18)
            f"{minResolution:.2G}",  # Minimum user-intended resolution (#19)
            f"{coordMax:.5f}",  # Approximate maximum coordinate value (#20)
            author,  # Name of author (#21)
            authorOrg,  # Author's Organization (#22)
            "11",  # IGES version number (#23)
            "0",  # Drafting code standard (#24)
        ]

        count = writeLines(handle, lines=lines, counter=1, marginText="G", margin=MARGIN1 - 1)

        return count

    @staticmethod
    def writeDirectoryCurve(handle: TextIO, paramLines: int, pCount: int, dCount: int) -> int:
        entityType = 126  # 126 is rational b-spline type

        # Determine the status number
        blankStatus = "00"  # 00 means entity is to be displayed
        entitySwitch = "00"  # 00 means entity is independent
        entityUseFlag = "00"  # 00 means full 3D geometry
        hierarchy = "01"  # 01 means no previous directory entry attributes will apply
        statusNumber = blankStatus + entitySwitch + entityUseFlag + hierarchy

        row1 = []
        row1.append(f"{entityType}")  # Entity type number (#1)
        row1.append(f"{pCount}")  # Parameter data pointer (#2)
        row1.append("0")  # Structure (#3)
        row1.append("0")  # Line font pattern, 1 is solid line (#4)
        row1.append("0")  # Entity level (#5)
        row1.append("0")  # Viewing options, 0 is equal visibility (#6)
        row1.append("0")  # Transformation matrix pointer, 0 is no transformations (#7)
        row1.append("0")  # Label display associativity, 0 is no associativity (#8)
        row1.append(statusNumber)  # Status number (#9)

        row1Text = ""
        for string in row1:
            row1Text += f"{string:>8}"

        row1Text += "D"
        row1Text += f"{dCount:>7}\n"
        dCount += 1

        handle.write(row1Text)

        row2 = []
        row2.append(f"{entityType}")  # Entity type number (#1)
        row2.append("0")  # Line weight number (#2)
        row2.append("0")  # Line color number (#3)
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
        row2Text += f"{dCount:>7}\n"
        dCount += 1

        handle.write(row2Text)

        return dCount

    @staticmethod
    def writeParametersCurve(handle: TextIO, geo: NURBSCurve, pCount: int, counter: int):
        entryCode = 126
        k = geo.nCtl - 1
        m = geo.degree
        n = k - m + 1
        a = n + (2 * m)

        nDim = geo.nDim - 1  # Adjust for homegenous coords
        prop1 = 0 if nDim > 2 else 1  # nonplanar (always assumed)

        # Check the first and last control points to determine if the curve is open or closed
        if np.allclose(geo.ctrlPnts[0], geo.ctrlPnts[-1], atol=1e-14):
            prop2 = 1
        else:
            prop2 = 0

        prop3 = 0  # Curve will always be rational
        prop4 = 0  # Non-periodic (always assumed for now)

        paramText = f"{pCount}P"

        # The first line of the parameter section is specially formatted
        lineText = f"{entryCode:>8d},{k:>7d},{m:>7d},{prop1:>7d},{prop2:>7d},{prop3:>7d},{prop4:>7d},"
        lineText += f"{paramText:>{MARGIN1 - len(lineText)}}{counter:>{MARGIN2 - MARGIN1}}\n"
        handle.write(lineText)
        counter += 1

        lines = []
        # Add the knot vector from index 7 to index 7+a (defined above)
        for knot in geo.knotVec:
            val = formatParameterValue(knot)
            lines.append(val)

        # Add the weights from index 8+a to 8+a+k (defined above)
        if geo.rational:
            weights = geo.weights
        else:
            weights = np.ones(geo.nCtl)  # Weights are array of ones of length nCtl

        for weight in weights:
            val = formatParameterValue(weight)
            lines.append(val)

        # Add the control points (x, y, z are written consecutively)
        # From index 9+a+k to 11+a+4*k (defined above)

        # Get the weighted control points if rational, else just get the regular control points
        # because the weights will all be equal to one
        ctrlPnts = geo.ctrlPntsW[:, :-1] if geo.rational else geo.ctrlPnts

        # We need to add a zero y-coordinate to planar curves
        if nDim <= 2:
            ctrlPnts = np.column_stack((ctrlPnts, np.zeros(geo.nCtl)))

        for i in range(geo.nCtl):
            for iDim in range(3):  # Data should always be 3-dimensional at this point
                val = formatParameterValue(ctrlPnts[i, iDim])
                lines.append(val)

        # Get the start and end parameter values (we know these will always be 0 to 1)
        lines.append(formatParameterValue(0))
        lines.append(formatParameterValue(1))

        # Write the planar normal vector (we dont use these so leave them blank)
        if prop1 == 1:
            lines.append(formatParameterValue(0))
            lines.append(formatParameterValue(0))
            lines.append(formatParameterValue(1))
        else:
            lines.append(formatParameterValue(0))
            lines.append(formatParameterValue(0))
            lines.append(formatParameterValue(0))

        counter = writeLines(handle, lines=lines, counter=counter, marginText=paramText, margin=PARAMMARGIN)

        return counter
