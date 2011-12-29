all:
	cd src;$(MAKE)
	cp src/libskl_flycapture.a ../../lib
#	cd Windows; ./generate_vcproj.rb > FlyCaptureBase.vcproj

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src;$(MAKE) debug
	cp src/libskl_flycapture_d.a ../../lib

clean:
	cd src;$(MAKE) clean
