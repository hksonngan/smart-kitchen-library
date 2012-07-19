/*!
 * @file ImplicitShapeModel.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jul/11
 * @date Last Change: 2012/Jul/18.
 */
#include "ImplicitShapeModel.h"
#include <cassert>
using namespace skl;

ISMEntries::ISMEntries(const cv::Size& max_dist_size,int roughness):_roughness(roughness),_max_dist_size(max_dist_size){
//	std::cerr << _max_dist_size.width << " x " << _max_dist_size.height << std::endl;
	int num_x = (_max_dist_size.width*2 + _roughness-1)/roughness;
	int num_y = (_max_dist_size.height*2 + _roughness-1)/roughness;
//	std::cerr << num_x << " x " << num_y << std::endl;
	index.assign(num_y,std::vector<std::list<size_t> >(
				num_x,std::list<size_t>()));

}
ISMEntries::~ISMEntries(){}

void ISMEntries::clear(){
	entries.clear();
	index.clear();
}

void ISMEntries::insert(const ISMEntry& entry){
	std::list<size_t> *list = this->_getIndex(entry.pt());
	std::list<size_t>::iterator pp;
	int i=0;
	for(pp=list->begin(); pp!=list->end(); pp++){
//		std::cerr << entry.class_response() << std::endl;
//		std::cerr << (*pp)->class_response() << std::endl;
//		std::cerr << i << "/" << list->size() << std::endl;i++;
		if(entries[*pp].isSameEvidence(entry)){
			entries[*pp].merge(entry);
			std::cerr << "merged at" << i << "/" << list->size() << std::endl;
			return;
		}
		i++;
	}
	entries.push_back(entry);
	list->push_back(entries.size()-1);

}

std::list<size_t> ISMEntries::getIndex(const cv::Point2f& pt,float roughness)const{
	int sx,sy,ex,ey;
	cv::Point2f temp(pt);
	temp.x -= roughness;
	temp.y -= roughness;
	getIndex(temp,sx,sy);
	temp.x = pt.x + roughness;
	temp.y = pt.y + roughness;
	getIndex(temp,ex,ey);
	std::list<size_t> dist;
	for(int y=sy; y<=ey; y++){
		for(int x=sx; x<=ex; x++){
			std::list<size_t> part = index[y][x];
			dist.merge(part);
		}
	}

	return dist;
}


std::list<size_t>* ISMEntries::_getIndex(const cv::Point2f& pt){
	int x,y;
	getIndex(pt,x,y);
//	std::cerr << "index: " << x << ", " << y << std::endl;
//	std::cerr << "idxsize: " << index[0].size() <<  "x " << index.size() << std::endl;
//	std::cerr << index[y][x].size() << std::endl;
/*	if(y<0||y>=index.size()){
		std::cerr << "index: " << x << ", " << y << std::endl;
		std::cerr << "idxsize: " << index[0].size() <<  "x " << index.size() << std::endl;
	}
	else if(x<0||x>=index[0].size()){
		std::cerr << "index: " << x << ", " << y << std::endl;
		std::cerr << "idxsize: " << index[0].size() <<  "x " << index.size() << std::endl;
	}
*/	return &index[y][x];
}

void ISMEntries::getIndex(const cv::Point2f& _pt, int& x, int& y)const{
	cv::Point2f pt = _pt;
	pt.x += _max_dist_size.width;
	pt.y += _max_dist_size.height;

	x = (int)pt.x / _roughness;
	if(x<0){
		x = 0;
	}
	else if((size_t)x>=index[0].size()){
		x = index[0].size()-1;
	}

	y = (int)pt.y / _roughness;
	if(y<0){
		y = 0;
	}
	else if((size_t)y>=index.size()){
		y = index.size()-1;
	}
}
