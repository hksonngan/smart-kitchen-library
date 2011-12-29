default:
	$(MAKE) all

debug:
	$(MAKE) debug_all

all: core opencv kinect flycapture

debug_all: core opencv kinect flycapture

core:
	cd Core;$(MAKE)
cored:
	cd Core;$(MAKE) debug
opencv:
	cd OpenCV;$(MAKE)
opencvd:
	cd OpenCV;$(MAKE) debug
kinect:
	cd Kinect;$(MAKE)
kinectd:
	cd Kinect;$(MAKE) debug
flycapture:
	cd FlyCapture;$(MAKE)
flycaptured:
	cd FlyCapture;$(MAKE) debug
clean:
	cd Core;$(MAKE) clean
	cd OpenCV;$(MAKE) clean
	cd Kinect;$(MAKE) clean
	cd FlyCapture;$(MAKE) clean
