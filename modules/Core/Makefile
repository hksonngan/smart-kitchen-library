all:
	cd src;$(MAKE)
	cp src/libskl_core.a ../../lib
	cd VisualStudioProject; ./generate_vcproj.rb > SKLCore.vcproj

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src;$(MAKE) debug
	cp src/libskl_core_d.a ../../lib

clean:
	cd src;$(MAKE) clean
