/*!
 * @file VideoCaptureImageList.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/18
 * @date Last Change: 2012/Jul/09.
 */
#include "VideoCaptureImageList.h"
#include <fstream>
using namespace skl;


/*!
 * @brief デフォルトコンストラクタ
 */
VideoCaptureImageList::VideoCaptureImageList():index(0),bayer_change(-1),imread_option(-1),checked_frame_pos(-1){
}

/*!
 * @brief デストラクタ
 */
VideoCaptureImageList::~VideoCaptureImageList(){

}

bool VideoCaptureImageList::open(const std::string& filename){
	release();
	std::ifstream fin;
	fin.open(filename.c_str());
	if(!fin){
		std::cerr << "ERROR: failed to open file '" << filename << "'." << std::endl;
		return false;
	}
	std::string str;
	while(fin && std::getline(fin,str)){
		img_list.push_back(str);
	}
	fin.close();
	return params.set(FRAME_COUNT,img_list.size());
}

void VideoCaptureImageList::release(){
	index = 0;
	img_list.clear();
	bayer_change = -1;
	imread_option = -1;
	checked_frame_pos = -1;
}

bool VideoCaptureImageList::grab(){
	if(static_cast<size_t>(index) >= img_list.size()) return false;
	checked_frame_pos = index;
	index++;
	return true;
}

bool VideoCaptureImageList::retrieve(cv::Mat& image, int channel){
	if(static_cast<size_t>(checked_frame_pos) >= img_list.size() || checked_frame_pos < 0){
		image.release();
		return false;
	}
	else if(bayer_change<=0){
		image = cv::imread(img_list[checked_frame_pos],imread_option);
	}
	else{
		cv::Mat bayer = cv::imread(img_list[checked_frame_pos],-1);
		if(bayer.channels()!=1){
			bayer.copyTo(image);
		}
		else{
			skl::cvtBayer2BGR(bayer,image,bayer_change,BAYER_EDGE_SENSE);
		}
	}
	return true;
}

bool VideoCaptureImageList::set(capture_property_t prop_id,double val){
	switch(prop_id){
		case POS_FRAMES:
			index = static_cast<int>(val);
			if(static_cast<size_t>(index) < img_list.size()){
				checked_frame_pos = index;
				return true;
			}
			else{
				return false;
			}
		case CONVERT_RGB:
			if(46 <= val && val < 50){
				bayer_change = static_cast<int>(val);
			}
			else if(val < 0){
				bayer_change = -1;
			}
			else{
				bayer_change = 0;
				imread_option = 1; // グレースケール画像でも3chで読み込む
			}
			return true;
		case MONOCROME:
			if(val>1){
				imread_option = 0;
				bayer_change = -1;
			}
			return true;
		default:
			return false;
	}
}

double VideoCaptureImageList::get(capture_property_t prop_id){
	switch(prop_id){
		case POS_FRAMES:
			return checked_frame_pos;
		case FRAME_COUNT:
			return img_list.size();
		case MONOCROME:
			if(imread_option==0){
				return 1.0;
			}
			else{
				return -1.0;
			}
		case CONVERT_RGB:
			return bayer_change;
		default:
			return 0.0;
	}
}
