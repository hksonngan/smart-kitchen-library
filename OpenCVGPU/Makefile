all:
	cd src/cuda;$(MAKE)
	cd src;$(MAKE)
	cp src/libskl_opencvgpu.a ../lib
#	cd VisualStudioProject; ./generate_vcproj.rb > SKLOpenCV.vcproj

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src/cuda;$(MAKE) debug
	cd src;$(MAKE) debug
	cp src/libskl_opencvgpu_d.a ../lib

clean:
	cd src;$(MAKE) clean
	cd src/cuda;$(MAKE) clean
