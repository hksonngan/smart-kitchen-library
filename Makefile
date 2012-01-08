default:
	$(MAKE) all

debug:
	$(MAKE) debug_all

all: core opencv

debug_all: core opencv

core:
	cd Core;$(MAKE)
cored:
	cd Core;$(MAKE) debug
opencv:
	cd OpenCV;$(MAKE)
opencvd:
	cd OpenCV;$(MAKE) debug
clean:
	cd Core;$(MAKE) clean
	cd OpenCV;$(MAKE) clean
