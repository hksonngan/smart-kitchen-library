/*!
 * @file FlyCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2012/Sep/24.
 */
#include "sklFlyCapture.h"

using namespace skl;

cv::Ptr<FlyCapture2::BusManager> FlyCapture::busMgr;
/*!
 * @brief デフォルトコンストラクタ
 */
FlyCapture::FlyCapture():is_started(false){
}

/*!
 * @brief デストラクタ
 */
FlyCapture::~FlyCapture(){

}

bool FlyCapture::open(){
	release();

	FlyCapture2::Error error;

#ifdef DEBUG
	FlyCapture2PrintBuildInfo();
#endif // DEBUG

	busMgr = new FlyCapture2::BusManager();
	unsigned int numCameras;
	error = busMgr->GetNumOfCameras(&numCameras);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
#ifdef DEBUG
	printf( "Number of flycapture cameras: %u\n", numCameras );
#endif // DEBUG
	if(numCameras==0) return false;

	cam_interface.resize(numCameras);
	fcam_interface.resize(numCameras);

	for(size_t i=0;i<numCameras;i++){
		VideoCaptureFlyCapture* capture = new VideoCaptureFlyCapture(busMgr);
		if(!capture->open(i)) return false;
		cam_interface[i] = capture;
		fcam_interface[i] = capture;
	}

	return true;
}

bool FlyCapture::sync_capture_start(FlyCapture2::Camera** ppCameras){
	if(size()==0) return false;
	FlyCapture2::Error error = FlyCapture2::Camera::StartSyncCapture((unsigned int)fcam_interface.size(), (const FlyCapture2::Camera**)ppCameras );
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
	for(size_t i=0;i<fcam_interface.size();i++){
		fcam_interface[i]->is_started = true;
	}

	is_started=true;
	return true;
}

void FlyCapture::release(){
	VideoCapture::release();

	if(!busMgr.empty()){
		busMgr.release();
	}
	is_started = false;
	return;
}

bool FlyCapture::grab(){
	if(!busMgr.empty() && !fcam_interface.empty() && !is_started){
		FlyCapture2::Camera** ppCameras = new FlyCapture2::Camera*[fcam_interface.size()];
		for(size_t i=0;i<fcam_interface.size();i++){
			ppCameras[i] = &(fcam_interface[i]->camera);
		}
		sync_capture_start(ppCameras);
		delete ppCameras;
	}
	return VideoCapture::grab();
}

