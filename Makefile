default: core opencv flycapture

debug: cored opencvd flycaptured

core:
	cd Core;$(MAKE)
cored:
	cd Core;$(MAKE) debug
opencv:
	cd OpenCV;$(MAKE)
opencvd:
	cd OpenCV;$(MAKE) debug
flycapture:
	cd FlyCapture;$(MAKE)
flycaptured:
	cd FlyCapture;$(MAKE) debug

clean:
	cd Core;$(MAKE) clean
	cd OpenCV;$(MAKE) clean
	cd FlyCapture;$(MAKE) clean
