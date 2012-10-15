/*!
 * @file VideoCaptureFlyCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/18
 * @date Last Change: 2012/Oct/01.
 */
#include "VideoCaptureFlyCapture.h"

using namespace skl;

std::map<capture_property_t, FlyCapture2::PropertyType> VideoCaptureFlyCapture::prop_type_map;

/*!
 * @brief デフォルトコンストラクタ
 */
VideoCaptureFlyCapture::VideoCaptureFlyCapture(cv::Ptr<FlyCapture2::BusManager> busMgr):is_opened(false),is_started(false),busMgr(busMgr){
	if(prop_type_map.empty()){
		initialize_prop_type_map();
	}
	set(skl::FPS, 15);
	set(skl::FRAME_WIDTH, 1024);
	set(skl::FRAME_HEIGHT, 680);
	set(skl::CONVERT_RGB,1);
}

/*!
 * @brief デストラクタ
 */
VideoCaptureFlyCapture::~VideoCaptureFlyCapture(){

}

/*!
 * @brief 画像読み込みには対応していないので呼ばれないprivate関数
 * */
bool VideoCaptureFlyCapture::open(const std::string& filename){
	return false;
}

/*!
 * @brief デバイス番号を指定するOpen
 * */
bool VideoCaptureFlyCapture::open(int device){
	FlyCapture2::Error error;
	FlyCapture2::PGRGuid guid;
	FlyCapture2::CameraInfo camInfo;

	if(busMgr.empty()) return false;
	error = busMgr->GetCameraFromIndex( device, &guid );
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	// Connect to cameras
	error = camera.Connect( &guid );
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	// Get the camera information
	error = camera.GetCameraInfo( &camInfo );
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

#ifdef _DEBUG
	std::cout << "\n*** INFORMATION FOR CAMERA " << device << " ***" << std::endl;
	FlyCapture2PrintCameraInfo(camInfo);
#endif
	error = camera.SetVideoModeAndFrameRate(
			getVideoMode(),
			getFrameRate());
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	// Set mode for Y8 returning image
	// 0x80000080 -> return by bayer array
	// 0x80000000 -> return by normal gray scale.
	int reg_val = 0x80000000;
	if(get(skl::CONVERT_RGB)>=0 || get(skl::MONOCROME)<=0){
		reg_val = 0x80000080;
	}
	error = camera.WriteRegister(0x1048,reg_val);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	FlyCapture2::TriggerDelay trigger_delay;
	trigger_delay.onOff = true;
	trigger_delay.valueA = 0;
	error = camera.SetTriggerDelay(
			&trigger_delay);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;


	return is_opened = set(CONVERT_RGB,1.0);
}

FlyCapture2::FrameRate VideoCaptureFlyCapture::getFrameRate(double fps){
	if(fps >= 240){
		return FlyCapture2::FRAMERATE_240;
	}
	if(fps >= 120){
		return FlyCapture2::FRAMERATE_120;
	}
	if(fps >= 60){
		return FlyCapture2::FRAMERATE_60;
	}
	if(fps >= 30){
		return FlyCapture2::FRAMERATE_30;
	}
	if(fps >= 15){
		return FlyCapture2::FRAMERATE_15;
	}
	if(fps >= 7.5){
		return FlyCapture2::FRAMERATE_7_5;
	}
	if(fps >= 3.75){
		return FlyCapture2::FRAMERATE_3_75;
	}
	return FlyCapture2::FRAMERATE_1_875;
}

#define caseVideoMode( _width, _height, suffix)\
	case _width:\
		assert(height==_height);\
		return FlyCapture2::VIDEOMODE_##_width##x##_height##suffix


FlyCapture2::VideoMode VideoCaptureFlyCapture::getVideoMode(int width,int height){
	switch(width){
		caseVideoMode(640,480,Y8);
		caseVideoMode(1024,768,Y8);
		caseVideoMode(1280,960,Y8);
		caseVideoMode(1600,1200,Y8);
	}
	return FlyCapture2::VIDEOMODE_640x480Y8;
}

void VideoCaptureFlyCapture::release(){
	is_opened = false;
	is_started = false;
	flycap_image.ReleaseBuffer();
	camera.StopCapture();
	camera.Disconnect();
}

bool VideoCaptureFlyCapture::grab(){
	if(!is_started){
		camera.StartCapture();
		is_started = true;
	}
	FlyCapture2::Error error = camera.RetrieveBuffer(&flycap_image);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
	return true;
}

bool VideoCaptureFlyCapture::retrieve(cv::Mat& image, int channels){
	FlyCapture2::PixelFormat pixFormat;
	FlyCapture2::BayerTileFormat bayerTileFormat;
	unsigned int rows,cols,stride;
	flycap_image.GetDimensions( &rows, &cols, &stride, &pixFormat, &bayerTileFormat);
	cv::Size size(cols,rows);
	if(size.height==0 || size.width==0) return false;

	if(get(skl::CONVERT_RGB)>=0){
		// create header to treat camImage[device] as cv::Mat
		cv::Mat bayer = cv::Mat(size,CV_8UC1,flycap_image.GetData(),stride);

		if(image.size()!=size || image.depth() != CV_8UC3 || !image.isContinuous()){
			image = cv::Mat(size,CV_8UC3);
		}
		assert(image.isContinuous());

		// get bayer pattern for cvtBayer2BGR
		int bayer_pattern;
		switch(bayerTileFormat){
			case FlyCapture2::RGGB:
				bayer_pattern = CV_BayerBG2BGR;
				break;
			case FlyCapture2::GRBG:
				bayer_pattern = CV_BayerGB2BGR;
				break;
			case FlyCapture2::GBRG:
				bayer_pattern = CV_BayerGR2BGR;
				break;
			case FlyCapture2::BGGR:
				bayer_pattern = CV_BayerRG2BGR;
				break;
			default:
				return false;
		}
		skl::cvtBayer2BGR(bayer,image,bayer_pattern,skl::BAYER_EDGE_SENSE);
	}
	else{
		if(image.size()!=size || image.depth() != CV_8UC1 || !image.isContinuous()){
			image = cv::Mat(size,CV_8UC1);
		}
		assert(image.isContinuous());
		assert(image.elemSize()*size.width*size.height == flycap_image.GetDataSize());
		memcpy(image.data,flycap_image.GetData(),flycap_image.GetDataSize());
	}

	return true;
}

bool VideoCaptureFlyCapture::set(capture_property_t prop_id,double val){
	if( isFlyCapProperty(prop_id) && isOpened() ){
		return set_flycap(prop_id,val);
	}
	return set_for_develop(prop_id,val);
}

bool VideoCaptureFlyCapture::set_for_develop(capture_property_t prop_id,double val){
	bool capture_color_image = true;

	switch(prop_id){
		case skl::CONVERT_RGB:
			if(val<=0) capture_color_image = false;
			break;
		case skl::MONOCROME:
			if(val>0) capture_color_image = false;
			break;
		default:
			return false;
	}

	FlyCapture2::Error error;
	if(capture_color_image){
		// Y8でBayerPattern画像を取得するために
		// Registerをいじる。
		error = camera.WriteRegister(0x1048,0x80000080);
		if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
		return params.set(prop_id,1.0) && params.set(skl::MONOCROME,-1.0);
	}
	else{
		error = camera.WriteRegister(0x1048,0x80000000);
		if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
		return params.set(prop_id,1.0) && params.set(skl::CONVERT_RGB,-1.0);
	}
}


bool VideoCaptureFlyCapture::set_flycap(capture_property_t prop_id,double val){
	std::map<capture_property_t, FlyCapture2::PropertyType>::iterator pp = prop_type_map.find(prop_id);
	if(prop_type_map.end()==pp) return false;
	FlyCapture2::PropertyInfo prop_info(pp->second);
	FlyCapture2::Error error = camera.GetPropertyInfo(&prop_info);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	FlyCapture2::Property prop(pp->second);
	camera.GetProperty(&prop);
	if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;

	if(val==static_cast<double>(DC1394_OFF)){
		if(prop_info.onOffSupported){
			prop.onOff = false;
			camera.SetProperty(&prop);
			if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
			return true;
		}
		return false;
	}
	else if(val==static_cast<double>(DC1394_MODE_AUTO)){
		if(prop_info.autoSupported){
			prop.autoManualMode = true;
			camera.SetProperty(&prop);
			if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
			return true;
		}
		return false;
	}
	else if(val==static_cast<double>(DC1394_MODE_ONE_PUSH_AUTO)){
		if(prop_info.onePushSupported){
			prop.onePush = true;
			camera.SetProperty(&prop);
			if(!SKL_FLYCAP2_CHECK_ERROR(error)) return false;
			return true;
		}
		return false;
	}

	// autoManualMode
	if(!prop_info.manualSupported){
		// you cannot set any value
		return false;
	}


	prop.type = pp->second;
	prop.autoManualMode = false;
	if(prop_info.onOffSupported){
		prop.onOff = true;
	}

	if(prop_info.absValSupported){
		try{
			if(val <= prop_info.absMin || prop_info.absMax <= val){
				std::stringstream ss;
				ss << "Warning: '" << val << "' is out of range for the property. ";
				ss << "Value must be in range [" << prop_info.absMin << ", " << prop_info.absMax << "].";
				throw(ss.str());
			}
		}
		catch(std::string error_message){
			std::cerr << error_message << std::endl;
			return false;
		}
		prop.absControl = true;
		prop.absValue = static_cast<float>(val);
	}
	else{
		try{
			if(val <= prop_info.min || prop_info.max <= val){
				std::stringstream ss;
				ss << "Warning: '" << val << "' is out of range for the property. ";
				ss << "Value must be in range [" << prop_info.min << ", " << prop_info.max << "].";
				throw(ss.str());
			}
		}
		catch(std::string error_message){
			std::cerr << error_message << std::endl;
			return false;
		}
		if(pp->second == FlyCapture2::WHITE_BALANCE){
			params.set(prop_id,val);
			prop.valueB = (unsigned int)params.get(skl::WHITE_BALANCE_BLUE_U);
			prop.valueA = (unsigned int)params.get(skl::WHITE_BALANCE_RED_V);
			if(prop.valueA == 0.0 || prop.valueB == 0.0) return true;
		}
		else{
			prop.valueA = (unsigned int)val;
		}
	}
	error = camera.SetProperty(&prop);
	return params.set(prop_id,val);
}

double VideoCaptureFlyCapture::get(capture_property_t prop_id){
	if(isFlyCapProperty(prop_id) && isOpened()){
		return get_flycap(prop_id);
	}
	return get_for_develop(prop_id);
}

double VideoCaptureFlyCapture::get_for_develop(capture_property_t prop_id){
	return params.get(prop_id);
}

double VideoCaptureFlyCapture::get_flycap(capture_property_t prop_id){

	std::map<capture_property_t, FlyCapture2::PropertyType>::iterator pp = prop_type_map.find(prop_id);
	if(prop_type_map.end()==pp) return 0.0;

	FlyCapture2::PropertyInfo prop_info(pp->second);
	FlyCapture2::Error error = camera.GetPropertyInfo(&prop_info);
	if(error.GetType() == FlyCapture2::PGRERROR_PROPERTY_NOT_PRESENT){
		return 0.0;
	}
	assert(SKL_FLYCAP2_CHECK_ERROR(error));

	FlyCapture2::Property prop(pp->second);
	error = camera.GetProperty(&prop);
	assert(SKL_FLYCAP2_CHECK_ERROR(error));

	double value;
	if(prop_info.absValSupported){
		value = prop.absValue;
	}
	else{
		if(pp->second != FlyCapture2::WHITE_BALANCE || prop_id == skl::WHITE_BALANCE_BLUE_U){
			value = prop.valueB;
		}
		else{
			assert(prop_id == skl::WHITE_BALANCE_RED_V);
			value = prop.valueA;
		}
	}
	if(value==0.0) return 0.0;
	if(!params.set(prop_id,value)) return 0.0;
	return value;
}

bool VideoCaptureFlyCapture::isFlyCapProperty(capture_property_t prop_id){
	return prop_type_map.end()!=prop_type_map.find(prop_id);
}


#define set_map(prop) \
	prop_type_map[skl::prop] = FlyCapture2::prop;

void VideoCaptureFlyCapture::initialize_prop_type_map(){
	set_map(BRIGHTNESS);
	set_map(AUTO_EXPOSURE);
	set_map(SHARPNESS);
	set_map(HUE);
	set_map(SATURATION);
	set_map(GAMMA);
	set_map(GAIN);
	prop_type_map[skl::TRIGGER] = FlyCapture2::TRIGGER_MODE;
	set_map(TRIGGER_DELAY);
	set_map(IRIS);
	set_map(FOCUS);
	set_map(ZOOM);
	set_map(PAN);
	set_map(TILT);
	set_map(SHUTTER);
	prop_type_map[skl::WHITE_BALANCE_BLUE_U] = FlyCapture2::WHITE_BALANCE;
	prop_type_map[skl::WHITE_BALANCE_RED_V] = FlyCapture2::WHITE_BALANCE;
}

