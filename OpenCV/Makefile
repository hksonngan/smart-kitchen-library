all:
	cd src;$(MAKE)
	cp src/libskl_opencv.a ../
#	cd VisualStudioProject; ./generate_vcproj.rb > SKLOpenCV.vcproj

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src;$(MAKE) debug
	cp src/libskl_opencv_d.a ../

clean:
	cd src;$(MAKE) clean
