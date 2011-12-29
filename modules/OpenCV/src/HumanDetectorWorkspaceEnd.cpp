#include "HumanDetectorWorkspaceEnd.h"
#include <set>

using namespace skl;

HumanDetectorWorkspaceEnd::HumanDetectorWorkspaceEnd(){}
HumanDetectorWorkspaceEnd::~HumanDetectorWorkspaceEnd(){}

void HumanDetectorWorkspaceEnd::setWorkspaceEnd(const cv::Mat& workspace_end){
	assert(CV_8UC1 == workspace_end.type());
	this->workspace_end = workspace_end;
}

std::list<size_t> HumanDetectorWorkspaceEnd::compute(
		const cv::Mat& src,
		const cv::Mat& mask,
		cv::Mat& human){
	assert(mask.size()==workspace_end.size());
	human = cv::Mat::zeros(mask.size(),CV_8UC1);
	std::set<size_t> human_regions;
	for(int y=0;y<mask.rows;y++){
		for(int x=0;x<mask.cols;x++){
			if(workspace_end.at<unsigned char>(y,x)==0) continue;
			if(mask.at<short>(y,x)==0) continue;
			human_regions.insert(mask.at<short>(y,x));
		}
	}

	if(human_regions.empty()){
		return std::list<size_t>();
	}

	std::list<size_t> _human_regions;
	CvMat _human = human;
	cv::Mat _temp = cv::Mat::zeros(mask.size(),CV_8UC1);
	CvMat temp = _temp;
	for(std::set<size_t>::iterator iter = human_regions.begin();
			iter != human_regions.end();iter++){
		_human_regions.push_back(*iter);
	}
	for(int y=0;y<mask.rows;y++){
		for(int x=0;x<mask.cols;x++){
			if(human_regions.end()!=human_regions.find(mask.at<short>(y,x))){
				human.at<unsigned char>(y,x) = 255;
			}
		}
	}


	return _human_regions;
}

