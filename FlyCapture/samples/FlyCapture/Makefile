# @file Makefile for sample_fly_capture
# @author a_hasimoto
OBJNAME = sample_fly_capture

OPT = 

# includeするファイルへのパス
INC =
# linkするライブラリへのパス
LIBS =
LIBS_OPT =
LIBS_DEBUG =

# コンパイラの設定
CC = g++
CFLAGS = -Wall
CFLAGS_D = -Wall

# skl用の設定
PATH_TO_SKL_ROOT = ../../../../
BINDIR = ${PATH_TO_SKL_ROOT}/bin/
ADDITIONAL_SKL_MODULE = OpenCV Core
ADDITIONAL_SKL_MODULEs = opencv core
SKL_LIBDIR = ${PATH_TO_SKL_ROOT}/lib
SKL_INCDIR += ${ADDITIONAL_SKL_MODULE:%=${PATH_TO_SKL_ROOT}modules/%/include/}
INC += ${SKL_INCDIR:%=-I %}
LIBS += -L${SKL_LIBDIR} ${ADDITIONAL_SKL_MODULEs:%=-lskltest_%}
LIBS_DEBUG += -L${SKL_LIBDIR} ${ADDITIONAL_SKL_MODULEs:%=-lskltest_%_d}

# public modules
ADDITIONAL_PUBLIC_SKL_MODULE = FlyCapture OpenCV Core
ADDITIONAL_PUBLIC_SKL_MODULEs = flycapture opencv core
PATH_TO_PUBLIC_SKL = ${PATH_TO_SKL_ROOT}/public_modules
INC += -I ${PATH_TO_PUBLIC_SKL}/include
PUBLIC_LIBDIR = ${PATH_TO_PUBLIC_SKL}/lib
LIBS += -L${PUBLIC_LIBDIR} ${ADDITIONAL_PUBLIC_SKL_MODULEs:%=-lskl_%}
LIBS_DEBUG += -L${PUBLIC_LIBDIR} ${ADDITIONAL_PUBLIC_SKL_MODULEs:%=-lskl_%}

# options for OpenCV
piriod:= .
empty:=
space:= $(empty) $(empty)
OPENCVVER	= $(subst $(piriod),$(space),$(shell pkg-config --modversion opencv))
INC += `pkg-config --cflags opencv`
LIBS_OPT += `pkg-config --libs opencv` -ltbb

CFLAGS += -DCV_MAJOR_VERSION=$(word 1,$(OPENCVVER))
CFLAGS += -DCV_MINOR_VERSION=$(word 2,$(OPENCVVER))
CFLAGS += -DHAVE_TBB
CFLAGS_D += -DCV_MAJOR_VERSION=$(word 1,$(OPENCVVER))
CFLAGS_D += -DCV_MINOR_VERSION=$(word 2,$(OPENCVVER))


# FlyCaptureの読み込み
INC += -I/usr/include/flycapture/
LIBS += -lflycapture
LIBS_DEBUG += -lflycapture




.SUFFIXES:.cpp

.PHONY:clean debug

default:${OBJNAME}
debug:${OBJNAME}_d

${OBJNAME}:${OBJNAME}.o  ${OPT:%=%.o}
	$(CC) ${CFLAGS} -o $@ $@.o ${OPT:%=%.o} $(INC) $(LIBS) $(LIBS_OPT) -O2 -D__linux__
	cp $@ $(BINDIR)$@

.cpp.o:
	$(CC) -c ${CFLAGS} -o $@ $? $(INC)

${OBJNAME}_d:${OBJNAME}_d.o  ${OPT:%=%_d.o}
	$(CC) ${CFLAGS} -o $@ $@.o ${OPT:%=%_d.o} $(INC) $(LIBS_DEBUG) $(LIBS_OPT) -DDEBUG

%_d.o:%.cpp
	$(CC) -g -c $(CFLAGS_D) -o $@ $? $(INC) -DDEBUG -D__linux_

clean:
	rm -rf $(OBJNAME).o $(OBJNAME)_d.o ${OPT:%=%.o} ${OPT:%=%_d.o} $(OBJNAME) $(OBJNAME)_d
	rm -rf $(BINDIR)$(OBJNAME)
