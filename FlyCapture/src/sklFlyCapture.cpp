/*!
 * @file FlyCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2013/Aug/15.
 */
#include "sklFlyCapture.h"

using namespace skl;

cv::Ptr<FlyCapture2::BusManager> FlyCapture::busMgr;
/*!
 * @brief デフォルトコンストラクタ
 */
FlyCapture::FlyCapture(bool _is_sync):is_started(false),is_sync(_is_sync){
	initialize();
}

/*!
 * @brief デストラクタ
 */
FlyCapture::~FlyCapture(){
}

void FlyCapture::initialize(){
	FlyCapture2::Error error;

#ifdef _DEBUG
	FlyCapture2PrintBuildInfo();
#endif // _DEBUG

	busMgr = new FlyCapture2::BusManager();
	unsigned int numCameras;
	error = busMgr->GetNumOfCameras(&numCameras);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return;
#ifdef _DEBUG
	printf( "Number of flycapture cameras: %u\n", numCameras );
#endif // _DEBUG

	cam_interface.resize(numCameras);
	fcam_interface.resize(numCameras);

	for(size_t i=0;i<numCameras;i++){
		VideoCaptureFlyCapture* capture = new VideoCaptureFlyCapture(busMgr);
		cam_interface[i] = capture;
		fcam_interface[i] = capture;
	}
#ifdef DEBUG
	FlyCapture2PrintBuildInfo();
#endif // DEBUG
}

bool FlyCapture::open(){
	if(isOpened()){
		release();
		initialize();
	}
	FlyCapture2::Error error;

	if(size()==0) return false;


	for(size_t i=0;i<size();i++){
		if(!fcam_interface[i]->open(i)) return false;
	}

	return true;
}

bool FlyCapture::async_capture_start(FlyCapture2::Camera** ppCameras){
	if(size()==0) return false;
	FlyCapture2::Error error;
	for(size_t i=0;i<(size_t)fcam_interface.size();i++){
		error = ppCameras[i]->StartCapture();
		if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
		fcam_interface[i]->is_started = true;
	}
	is_started = true;
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
		if(is_sync){
			sync_capture_start(ppCameras);
		}
		else{
			async_capture_start(ppCameras);
		}
		delete ppCameras;
	}
	return VideoCapture::grab();
}

