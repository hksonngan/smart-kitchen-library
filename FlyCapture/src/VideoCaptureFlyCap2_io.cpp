#include "VideoCaptureFlyCap2.h"
#include <sstream>
#include <iostream>

using namespace skl;
#define CHECK_ERROR(error) checkError(error, __FILE__, __LINE__)

std::map<capture_property_t, FlyCapture2::PropertyType> FlyCaptureCameraInterface::prop_type_map;

FlyCaptureCameraInterface::FlyCaptureCameraInterface(FlyCapture* fly_capture, int cam_id):VideoCaptureCameraInterface(fly_capture,cam_id),fly_capture(fly_capture){
	if(prop_type_map.empty()){
		initialize_prop_type_map();
	}
}

bool FlyCaptureCameraInterface::set(capture_property_t prop_id,double val){
	if(isFlyCapProperty(prop_id)){
		return set_flycap(prop_id,val);
	}
	return set_for_develop(prop_id,val);

}

bool FlyCaptureCameraInterface::set_for_develop(capture_property_t prop_id,double val){
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
		error = fly_capture->ppCameras[cam_id]->WriteRegister(0x1048,0x80000080);
		if(!fly_capture->CHECK_ERROR(error)) return false;
		return params.set(prop_id,1.0) && params.set(skl::MONOCROME,-1.0);
	}
	else{
		error = fly_capture->ppCameras[cam_id]->WriteRegister(0x1048,0x80000000);
		if(!fly_capture->CHECK_ERROR(error)) return false;
		return params.set(prop_id,1.0) && params.set(skl::CONVERT_RGB,-1.0);
	}
}


bool FlyCaptureCameraInterface::set_flycap(capture_property_t prop_id,double val){
	std::map<capture_property_t, FlyCapture2::PropertyType>::iterator pp = prop_type_map.find(prop_id);
	if(prop_type_map.end()==pp) return false;
	FlyCapture2::PropertyInfo prop_info(pp->second);
	FlyCapture2::Error error = fly_capture->ppCameras[cam_id]->GetPropertyInfo(&prop_info);
	if(!fly_capture->CHECK_ERROR(error)) return false;

	FlyCapture2::Property prop(pp->second);
	fly_capture->ppCameras[cam_id]->GetProperty(&prop);
	if(!fly_capture->CHECK_ERROR(error)) return false;

	if(val==static_cast<double>(DC1394_OFF)){
		if(prop_info.onOffSupported){
			prop.onOff = false;
			fly_capture->ppCameras[cam_id]->SetProperty(&prop);
			if(!fly_capture->CHECK_ERROR(error)) return false;
			return true;
		}
		return false;
	}
	else if(val==static_cast<double>(DC1394_MODE_AUTO)){
		if(prop_info.autoSupported){
			prop.autoManualMode = true;
			fly_capture->ppCameras[cam_id]->SetProperty(&prop);
			if(!fly_capture->CHECK_ERROR(error)) return false;
			return true;
		}
		return false;
	}
	else if(val==static_cast<double>(DC1394_MODE_ONE_PUSH_AUTO)){
		if(prop_info.onePushSupported){
			prop.onePush = true;
			fly_capture->ppCameras[cam_id]->SetProperty(&prop);
			if(!fly_capture->CHECK_ERROR(error)) return false;
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
			prop.valueB = params.get(skl::WHITE_BALANCE_BLUE_U);
			prop.valueA = params.get(skl::WHITE_BALANCE_RED_V);
			if(prop.valueA == 0.0 || prop.valueB == 0.0) return true;
		}
		else{
			prop.valueA = val;
		}
	}
	error = fly_capture->ppCameras[cam_id]->SetProperty(&prop);
	return params.set(prop_id,val);
}

double FlyCaptureCameraInterface::get(capture_property_t prop_id){
	if(isFlyCapProperty(prop_id)){
		return get_flycap(prop_id);
	}
	return get_for_develop(prop_id);
}

double FlyCaptureCameraInterface::get_for_develop(capture_property_t prop_id){
	return params.get(prop_id);
}

double FlyCaptureCameraInterface::get_flycap(capture_property_t prop_id){
	if(fly_capture->numCameras <= static_cast<unsigned int>(cam_id)) return 0.0;

	std::map<capture_property_t, FlyCapture2::PropertyType>::iterator pp = prop_type_map.find(prop_id);
	if(prop_type_map.end()==pp) return 0.0;

	FlyCapture2::PropertyInfo prop_info(pp->second);
	FlyCapture2::Error error = fly_capture->ppCameras[cam_id]->GetPropertyInfo(&prop_info);
	if(error == FlyCapture2::PGRERROR_PROPERTY_NOT_PRESENT){
		return 0.0;
	}
	assert(fly_capture->CHECK_ERROR(error));

	FlyCapture2::Property prop(pp->second);
	error = fly_capture->ppCameras[cam_id]->GetProperty(&prop);
	assert(fly_capture->CHECK_ERROR(error));

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


#define set_map(prop) \
	prop_type_map[skl::prop] = FlyCapture2::prop;

void FlyCaptureCameraInterface::initialize_prop_type_map(){
	prop_type_map[skl::FPS] = FlyCapture2::FRAME_RATE;
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

bool FlyCaptureCameraInterface::isFlyCapProperty(capture_property_t prop_id){
	return prop_type_map.end()!=prop_type_map.find(prop_id);
}

