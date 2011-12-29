all:
	cd src;$(MAKE)
	cp src/libskl_opencv.a ../../lib
#	cd Windows; ./generate_vcproj.rb > OpenCVBase.vcproj

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src;$(MAKE) debug
	cp src/libskl_opencv_d.a ../../lib

clean:
	cd src;$(MAKE) clean
