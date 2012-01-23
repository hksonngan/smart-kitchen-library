OBJS=$(patsubst %.cpp,%.o,$(shell find ./ -name "*.cpp" -print))

LIBNAME = libskl_boost

ADDITIONALMODULE = 

L_BITS := $(shell getconf LONG_BIT)

CC = g++


INC += -I../include -I../../include
INC += ${ADDITIONALMODULE:%=-I../../%/include/}

.SUFFIXES:.cpp

.PHONY:clean debug

default:${LIBNAME}.a

debug:${LIBNAME}_d.a

${LIBNAME}.a:${OBJS:%=${LIBNAME}.a(%)}

${LIBNAME}(%.o):%.o

${LIBNAME}_d.a:${OBJS:%.o=${LIBNAME}_d.a(%_d.o)}

${LIBNAME}_d(%_d.o):%_d.o

LIBS += -lboost_regex -lboost_tokenizer

# hoge.cpp -> hoge.o
.cpp.o:
	$(CC) -O2 -c $(CFLAGS) -o $@ $< $(INC) -D__linux__ $(LIBS)

# hoge.cpp -> hoge_d.o
%_d.o:%.cpp
	$(CC) -g -c $(CFLAGS) -o $@ $< $(INC) -DDEBUG -D__linux__ $(LIBS)

clean:
	rm -f *.o
	rm -f $(LIBNAME).a
	rm -f $(LIBNAME)_d.a
