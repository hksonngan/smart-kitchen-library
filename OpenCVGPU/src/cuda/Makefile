CUFILES=$(shell find ./ -name "*.cu" -print)
LIBNAME ?= libskl_opencvgpu
STATIC_LIB = ${LIBNAME}.a

SRCDIR = ./
ROOTDIR = ..
ROOTBINDIR = ../
BINDIR = ../
ROOTOBJDIR = ./

INCLUDES += `pkg-config --cflags opencv`

include ./common.mk

.PHONY:debug
debug:
	make dbg=1 LIBNAME=libskl_opencvgpu_d
