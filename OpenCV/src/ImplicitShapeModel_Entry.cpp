/*!
 * @file ImplicitShapeModel.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jul/11
 * @date Last Change: 2012/Jul/18.
 */
#include "ImplicitShapeModel.h"
#include <cassert>
using namespace skl;

ISMEntry::ISMEntry(
		const cv::Point2f& pt,
		const std::vector<float>& scale,
		int class_response,
		const cv::Mat& patch){
	_pt.x = pt.x / scale[0];
	_pt.y = pt.y / scale[1];
	_class_response = class_response;

	if(patch.rows > 0 && patch.cols > 0){
		assert(patch.type()==CV_32FC1 || patch.type()==CV_8UC1);
		cv::Mat temp = patch;
		// adjust size to a square shape.
		if(patch.cols!=patch.rows && scale[0]!=scale[1]){
			cv::Size size(patch.size());
			if(scale[0]>scale[1]){
				assert(patch.cols>patch.rows);
				size.height = size.width;
			}
			else{
				assert(patch.rows>patch.cols);
				size.width = size.height;
			}
			cv::resize(patch,temp,size,0,0,cv::INTER_CUBIC);
		}

		// adjust elem_type to float
		if(temp.type()==CV_32FC1){
			_patch = temp.clone();
		}
		else if(temp.type()==CV_8UC1){
			temp.convertTo(_patch,CV_32FC1,1.0f/255);
		}
	}
}

ISMEntry::ISMEntry(const ISMEntry& other){
	_pt = other.pt();
	_class_response = other.class_response();
	_patch = other.patch().clone();
}

ISMEntry::~ISMEntry(){

}

bool ISMEntry::isSameEvidence(const ISMEntry& other)const{
//	std::cerr << _class_response << std::endl;
	if(_class_response != other.class_response()) return false;
	if(_pt.x != other.pt().x) return false;
	if(_pt.y != other.pt().y) return false;
	return true;
}

bool ISMEntry::merge(const ISMEntry& other){
	if(!isSameEvidence(other)) return false;
	if(_patch.rows == other.patch().rows) return false;
	if(_patch.cols == other.patch().cols) return false;
	_patch = (_patch + other.patch()) / 2;
	return true;
}

void ISMEntry::write(std::ostream& out)const{
	float fdata[2];
	fdata[0] = _pt.x;
	fdata[1] = _pt.y;
	out.write((char*)fdata,2*sizeof(float));
	out.write((char*)&_class_response,sizeof(int));
	int patch_size[2];
	patch_size[0] = _patch.cols;
	patch_size[1] = _patch.rows;
	out.write((char*)patch_size,2*sizeof(int));
	if(_patch.rows>0&&_patch.cols>0){
		for(int y=0;y<_patch.rows;y++){
			out.write((char*)_patch.ptr<float>(y),_patch.cols*sizeof(float));
		}
	}
}

void ISMEntry::read(std::istream& in){
	float fdata[2];
	in.read((char*)fdata,2*sizeof(float));
	_pt.x = fdata[0];
	_pt.y = fdata[1];
	in.read((char*)&_class_response,sizeof(int));
//	std::cerr << _pt.x << ", " << _pt.y << ": " << _class_response << std::endl;
	int patch_size[2];
	in.read((char*)patch_size,2*sizeof(int));
	if(patch_size[0]>0&&patch_size[1]>0){
		_patch = cv::Mat(cv::Size(patch_size[0],patch_size[1]),CV_32FC1);
		for(int y=0;y<_patch.rows;y++){
			in.read((char*)_patch.ptr<float>(y),_patch.cols*sizeof(float));
		}
	}
}