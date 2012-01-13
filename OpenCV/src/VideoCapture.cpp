/*!
 * @file VideoCapture.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2012/Jan/13.
 */
#include "VideoCapture.h"
#include <sstream>
#include <fstream>



using namespace skl;


#define set_name_id_map( prop_name ) property_name_id_map[#prop_name] = CV_CAP_PROP_##prop_name
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
	//	set_name_id_map(WHITE_BALANCE); // reserved parameter by OpenCV
	set_name_id_map(RECTIFICATION);
}

VideoParams::~VideoParams(){}
VideoParams::VideoParams(const std::string& filename){load(filename);}
VideoParams::VideoParams(const VideoParams& other){
	property_name_id_map = other.property_name_id_map;
	property_id_value_map = other.property_id_value_map;
}

bool VideoParams::set(const std::string& prop_name,double val){
	std::map<std::string,int>::iterator pp = property_name_id_map.find(prop_name);
	if(property_name_id_map.end()==pp) return false;
	return set(pp->second,val);
}
bool VideoParams::set(int prop_id,double val){
	property_id_value_map[prop_id] = val;
	return true;
}

VideoParamIter VideoParams::begin()const{return property_id_value_map.begin();}

VideoParamIter VideoParams::end()const{return property_id_value_map.end();}

std::string VideoParams::print()const{
	std::stringstream ss;
	for(std::map<std::string,int>::const_iterator iter = property_name_id_map.begin();
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
		std::map<std::string,int>::iterator pp = property_name_id_map.find(key_val[0]);
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
/*!
 * @brief デフォルトコンストラクタ
 */
VideoCapture::VideoCapture(){

}

/*!
 * @brief デストラクタ
 */
VideoCapture::~VideoCapture(){

}

bool VideoCapture::set(const VideoParams& params){
	for(VideoParamIter iter = params.begin();
			iter != params.end();iter++){
		if(!this->set(iter->first,iter->second)) return false;
	}
	return true;
}
bool VideoCapture::set(int prop_id,double val){
	return params.set(prop_id,val) && cv::VideoCapture::set(prop_id,val);
}

VideoParams VideoCapture::get(){
	for(std::map<std::string,int>::const_iterator iter = params.property_name_id_map.begin();
			iter!= params.property_name_id_map.end();iter++){
		double val = cv::VideoCapture::get(iter->second);
		if(val==0)continue;
		params.set(iter->second,val);
	}
	return params;
}


