# @file Makefile for sample_conffile_parser
# @author a_hasimoto
OBJNAME = sample_conffile_parser

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

# skl用の設定
PATH_TO_SKL_ROOT = ../../../../
BINDIR = ${PATH_TO_SKL_ROOT}/bin/
ADDITIONAL_SKL_MODULE = Core
ADDITIONAL_SKL_MODULEs = core
SKL_LIBDIR = ${PATH_TO_SKL_ROOT}/lib
SKL_INCDIR += ${ADDITIONAL_SKL_MODULE:%=${PATH_TO_SKL_ROOT}modules/%/include/}
INC += ${SKL_INCDIR:%=-I %}
LIBS += -L${SKL_LIBDIR} ${ADDITIONAL_SKL_MODULEs:%=-lskl_%}
LIBS_DEBUG += -L${SKL_LIBDIR} ${ADDITIONAL_SKL_MODULEs:%=-lskl_%_d}



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
	$(CC) -g -c $(CFLAGS) -o $@ $? $(INC) -DDEBUG -D__linux_

clean:
	rm -rf $(OBJNAME).o $(OBJNAME)_d.o ${OPT:%=%.o} ${OPT:%=%_d.o} $(OBJNAME) $(OBJNAME)_d
	rm -rf $(BINDIR)$(OBJNAME)
