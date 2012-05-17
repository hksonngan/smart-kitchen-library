default: all

all: core boost opencv flycapture opencvgpu

debug: cored boostd opencvd flycaptured opencvgpud

core:
	cd Core;$(MAKE)
cored:
	cd Core;$(MAKE) debug

boost:
	cd Boost;$(MAKE)
boostd:
	cd Boost;$(MAKE) debug

opencv:
	cd OpenCV;$(MAKE)
opencvd:
	cd OpenCV;$(MAKE) debug

flycapture:
	cd FlyCapture;$(MAKE)
flycaptured:
	cd FlyCapture;$(MAKE) debug

opencvgpu:
	cd OpenCVGPU;$(MAKE)
opencvgpud:
	cd OpenCVGPU;$(MAKE) debug

clean:
	cd Core;$(MAKE) clean
	cd OpenCV;$(MAKE) clean
	cd FlyCapture;$(MAKE) clean
	cd Boost;$(MAKE) clean
	cd OpenCVGPU;$(MAKE) clean
