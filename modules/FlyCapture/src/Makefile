OBJS=$(patsubst %.cpp,%.o,$(shell find ./ -name "*.cpp" -print))

LIBNAME = libskl_flycapture

ADDITIONALMODULE = OpenCV Core

BOMCHECK_SCRIPT = ../../../support/CheckAndAddBOM.rb

L_BITS := $(shell getconf LONG_BIT)

CC = g++

# options for OpenCV
piriod:= .
empty:=
space:= $(empty) $(empty)
OPENCVVER	= $(subst $(piriod),$(space),$(shell pkg-config --modversion opencv))
INC += `pkg-config --cflags opencv`
LIBS_OPT += `pkg-config --libs opencv`

CFLAGS += -DCV_MAJOR_VERSION=$(word 1,$(OPENCVVER))
CFLAGS += -DCV_MINOR_VERSION=$(word 2,$(OPENCVVER))

INC += -I../include -I../../../include
INC += ${ADDITIONALMODULE:%=-I../../%/include/}

.SUFFIXES:.cpp

.PHONY:clean debug

default:${LIBNAME}.a

debug:${LIBNAME}_d.a

${LIBNAME}.a:${OBJS:%=${LIBNAME}.a(%)}

${LIBNAME}(%.o):%.o

${LIBNAME}_d.a:${OBJS:%.o=${LIBNAME}_d.a(%_d.o)}

${LIBNAME}_d(%_d.o):%_d.o

# hoge.cpp -> hoge.o
.cpp.o:
	$(CC) -O2 -c $(CFLAGS) -o $@ $< $(INC) -D__linux__

# hoge.cpp -> hoge_d.o
%_d.o:%.cpp
	ruby $(BOM_SCRIPT) $@
	$(CC) -g -c $(CFLAGS) -o $@ $< $(INC) -DDEBUG -D__linux__

clean:
	rm -f *.o
	rm -f $(LIBNAME).a
	rm -f $(LIBNAME)_d.a
