/*!
 * @file VideoCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2012/Jan/17.
 */
#include "VideoCapture.h"
#include <sstream>
#include <fstream>

using namespace skl;

/********* VideoParams *********/
std::map<std::string,capture_property_t> VideoParams::property_name_id_map;

#define set_name_id_map( prop_name ) property_name_id_map[#prop_name] = prop_name
VideoParams::VideoParams(){
	set_name_id_map(POS_MSEC);
	set_name_id_map(POS_FRAMES);
	set_name_id_map(POS_AVI_RATIO);
	set_name_id_map(FRAME_WIDTH);
	set_name_id_map(FRAME_HEIGHT);
	set_name_id_map(FPS);
	set_name_id_map(FOURCC);
	set_name_id_map(FRAME_COUNT);
	set_name_id_map(FORMAT);
	set_name_id_map(MODE);
	set_name_id_map(BRIGHTNESS);
	set_name_id_map(CONTRAST);
	set_name_id_map(SATURATION);
	set_name_id_map(HUE);
	set_name_id_map(GAIN);
	set_name_id_map(EXPOSURE);
	set_name_id_map(CONVERT_RGB);
	set_name_id_map(WHITE_BALANCE_BLUE_U); // reserved parameter by OpenCV
	set_name_id_map(RECTIFICATION);
	set_name_id_map(MONOCROME);
	set_name_id_map(SHARPNESS);
	set_name_id_map(AUTO_EXPOSURE);
	set_name_id_map(GAMMA);
	set_name_id_map(TEMPERATURE);
	set_name_id_map(TRIGGER);
	set_name_id_map(TRIGGER_DELAY);
	set_name_id_map(WHITE_BALANCE_RED_V);
}
VideoParams::~VideoParams(){}
VideoParams::VideoParams(const std::string& filename){load(filename);}
VideoParams::VideoParams(const VideoParams& other){
	property_name_id_map = other.property_name_id_map;
	property_id_value_map = other.property_id_value_map;
}

bool VideoParams::set(const std::string& prop_name,double val){
	std::map<std::string,capture_property_t>::iterator pp = property_name_id_map.find(prop_name);
	if(property_name_id_map.end()==pp) return false;
	return set(pp->second,val);
}
bool VideoParams::set(capture_property_t prop_id,double val){
	property_id_value_map[prop_id] = val;
	return true;
}

double VideoParams::get(const std::string& prop_name)const{
	std::map<std::string,capture_property_t>::iterator pp = property_name_id_map.find(prop_name);
	if(property_name_id_map.end()==pp) return 0.0;
	return get(pp->second);
}

double VideoParams::get(capture_property_t prop_id)const{
	VideoParamIter pp = property_id_value_map.find(prop_id);
	if(property_id_value_map.end() == pp){
		return 0.0;
	}
	return pp->second;
}

VideoParamIter VideoParams::begin()const{return property_id_value_map.begin();}

VideoParamIter VideoParams::end()const{return property_id_value_map.end();}

std::string VideoParams::print()const{
	std::stringstream ss;
	for(std::map<std::string,capture_property_t>::const_iterator iter = property_name_id_map.begin();
			iter != property_name_id_map.end();iter++){
		VideoParamIter pval = property_id_value_map.find(iter->second);
		if(property_id_value_map.end()==pval)continue;
		ss << iter->first << ": " << pval->second << std::endl;
	}
	return ss.str();
}

bool VideoParams::scan(const std::string& str){
	std::vector<std::string> buf = split(str,"\n");
	for(size_t i=0;i<buf.size();i++){
		std::vector<std::string> key_val = split(buf[i],":");
		if(key_val.size()!=0) continue;
		std::map<std::string,capture_property_t>::iterator pp = property_name_id_map.find(key_val[0]);
		if( !set( key_val[0], atof(key_val[1].c_str()) ) ){
			return false;
		}
	}
	return true;
}

bool VideoParams::load(const std::string& filename){
	std::map<std::string,std::string> param_map;
	if(!parse_conffile(filename,param_map,":")) return false;
	for(std::map<std::string,std::string>::iterator iter = param_map.begin();
			iter != param_map.end();iter++){
		assert(set(iter->first,atof(iter->second.c_str())));
	}
	return true;
}

void VideoParams::save(const std::string& filename)const{
	std::ofstream fout;
	fout.open(filename.c_str());
	fout << print();
	fout.close();
}

/********** VideoCaptureCameraInterFace *********/
VideoCaptureCameraInterface::VideoCaptureCameraInterface(VideoCapture* video_capture,int cam_id):video_capture(video_capture),cam_id(cam_id){}
VideoCaptureCameraInterface::~VideoCaptureCameraInterface(){}

/*** Param IO ***/
bool VideoCaptureCameraInterface::set(const VideoParams& params){
	for(VideoParamIter iter = params.begin();
			iter != params.end();iter++){
		if(!this->set(iter->first,iter->second)) return false;
	}
	return true;
}

bool VideoCaptureCameraInterface::set(const std::string& prop_name, double val){
	std::map<std::string,capture_property_t>::const_iterator pp = params.property_name_id_map.find(prop_name);
	if(params.property_name_id_map.end() == pp) return false;
	return set(pp->second,val);
}

bool VideoCaptureCameraInterface::set(capture_property_t prop_id,double val){
	return params.set(prop_id,val) && video_capture->cv::VideoCapture::set(prop_id,val);
}

bool VideoCaptureCameraInterface::set(const std::string& prop_name, camera_mode_t mode){
	std::map<std::string,capture_property_t>::const_iterator pp = params.property_name_id_map.find(prop_name);
	if(params.property_name_id_map.end() == pp) return false;
	return set(pp->second,mode);
}

bool VideoCaptureCameraInterface::set(capture_property_t prop_id,camera_mode_t mode){
	double val = params.get(prop_id);
	bool success = set(prop_id, static_cast<double>(mode));
	params.set(prop_id,val);
	return success;
}

const VideoParams& VideoCaptureCameraInterface::get(){
	for(std::map<std::string,capture_property_t>::const_iterator iter = params.property_name_id_map.begin();
			iter!= params.property_name_id_map.end();iter++){
		this->get(iter->second);
	}
	return params;
}

double VideoCaptureCameraInterface::get(const std::string& prop_name){
	std::map<std::string,capture_property_t>::const_iterator pp =  params.property_name_id_map.find(prop_name);
	if(params.property_name_id_map.end() == pp) return 0;
	return get(pp->second);
}


double VideoCaptureCameraInterface::get(capture_property_t prop_id){
	double val =  video_capture->cv::VideoCapture::get(prop_id);
	if(val==0.0) return 0.0;
	params.set(prop_id,val);
	return val;
}

/*!
 * @brief カメラ毎に別々に画像を取り出すクラス。ただし、VideoCaptureの>>と違って、VideoCapture::grab()は別途実行すること
 * */
VideoCaptureCameraInterface& VideoCaptureCameraInterface::operator>> (cv::Mat& image){
	video_capture->retrieve(image,cam_id+1);
	return *this;
}

/******** VideoCapture ********/
/*!
 * @brief デフォルトコンストラクタ
 */
VideoCapture::VideoCapture(){
	cam_interface.assign(1,NULL);
	cam_interface[0] = new VideoCaptureCameraInterface(this);
}

/*!
 * @brief デストラクタ
 */
VideoCapture::~VideoCapture(){
	delete cam_interface[0];
}

bool VideoCapture::set(const VideoParams& params){
	assert(cam_interface.size()>0);
	return cam_interface[0]->set(params);
}
bool VideoCapture::set(capture_property_t prop_id,double val){
	assert(cam_interface.size()>0);
	return cam_interface[0]->set(prop_id,val);
}

bool VideoCapture::set(const std::string& prop_name, double val){
	assert(cam_interface.size()>0);
	return cam_interface[0]->set(prop_name,val);
}
bool VideoCapture::set(capture_property_t prop_id,camera_mode_t mode){
	assert(cam_interface.size()>0);
	return cam_interface[0]->set(prop_id,mode);
}

bool VideoCapture::set(const std::string& prop_name,camera_mode_t mode){
	assert(cam_interface.size()>0);
	return cam_interface[0]->set(prop_name,mode);
}



const VideoParams& VideoCapture::get(){
	assert(cam_interface.size()>0);
	return cam_interface[0]->get();
}


double VideoCapture::get(const std::string& prop_name){
	assert(cam_interface.size()>0);
	return cam_interface[0]->get(prop_name);
}

double VideoCapture::get(capture_property_t prop_id){
	assert(cam_interface.size()>0);
	return cam_interface[0]->get(prop_id);
}

/*
 * @brief this is used to set/get parameters for cameras independently, when VideoCapture manage several cameras.
 * @param device derect camera device.
 * @return an instance which has the same get/set functions, but for camera[device].
 * */
VideoCaptureCameraInterface& VideoCapture::operator[](int device){
	assert(static_cast<unsigned int>(device) < cam_interface.size());
	return *(cam_interface[device]);
}
