#*********************************************************
# Makefile for pySpline
# G. Kenway May 21, 2009
#*********************************************************

MAIN_DIR = ../
COMMON_FILE = $(MAIN_DIR)/Common.mk
RULES_FILE  = $(MAIN_DIR)/Rules.mk
include ${COMMON_FILE}
include ${RULES_FILE}
vpath %.o $(OBJDIR)

# Name of files to be compiled:
MODS = precision.o

OBJS =  basis.o\
	findSpan.o\
	evalCurve.o\
	evalCurveNURBS.o\
	evalSurfaceNURBS.o\
	evalSurface.o\
	multiplicity.o\
	binomialCoeff.o\
	curveFitting.o\
	surfaceFitting.o\
	projections.o\
	linearAlgebra.o\
	newtonSolver.o\
	pointInversion.o\
	optimizer.o\

default:
	$(MAKE) $(MODS)
	cp -f *.mod ../mod
	$(MAKE) $(OBJS)

clean:
	@echo "        Making clean in src... "
	rm -f $(MAKE_CLEAN_ARGUMENTS)
