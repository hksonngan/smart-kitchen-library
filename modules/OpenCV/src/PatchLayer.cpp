/*!
 * @file PatchLayer.cpp
 * @author ∂∂À‹∆ÿªÀ
 * @date Last Change:2011/Oct/23.
 */

#include "PatchModel.h"
using namespace mmpl;
using namespace mmpl::image;

PatchLayer::PatchLayer(std::map<size_t,Patch>* patches):patches(patches){

}

PatchLayer::~PatchLayer(){

}

void PatchLayer::push(size_t ID){
	layer_order.push_back(ID);
}

void PatchLayer::erase(size_t ID){
	std::list<size_t>::iterator ptar = std::find(
			layer_order.begin(),
			layer_order.end(),
			ID);
	assert(layer_order.end()!=ptar);
	layer_order.erase(ptar);
}

size_t PatchLayer::getUpperPatch(size_t ID, Patch::Type type)const{
	std::list<size_t>::const_iterator ptar_id = std::find(
			layer_order.begin(),
			layer_order.end(),
			ID);
	assert(layer_order.end()!=ptar_id);
	ptar_id++;

	std::map<size_t,Patch>::const_iterator psrc,ptar;
	psrc = patches->find(ID);
	assert(patches->end()!=psrc);
	CvRect src_rect = psrc->second.getRect(type);

	for(;ptar_id!=layer_order.end();ptar_id++){
		ptar = patches->find(*ptar_id);
		assert(patches->end()!=ptar);
		CvRect tar_rect = ptar->second.getRect(type);

		CvRect common_rect = Patch::getCommonRectanglarRegion(src_rect,tar_rect);
		if(common_rect.width <= 0 || common_rect.height <= 0 ) continue;

		if(isOverlayed(common_rect,psrc->second,ptar->second, type)){
			return *ptar_id;
		}
	}
	return UINT_MAX;
}

std::vector<size_t> PatchLayer::getAllBeneathPatch(size_t ID,Patch::Type type){
	std::list<size_t>::const_reverse_iterator ptar_id = std::find(
			layer_order.rbegin(),
			layer_order.rend(),
			ID);
	assert(layer_order.rend()!=ptar_id);
	ptar_id++;

	std::vector<size_t> dst;
	std::map<size_t,Patch>::const_iterator psrc,ptar;
	psrc = patches->find(ID);
	assert(patches->end()!=psrc);
	CvRect src_rect = psrc->second.getRect(type);
	Image mask = psrc->second.getMask(type);

	for(;ptar_id!=layer_order.rend();ptar_id++){
		ptar = patches->find(*ptar_id);
		assert(patches->end()!=ptar);

		CvRect tar_rect = ptar->second.getRect(type);

		CvRect common_rect = Patch::getCommonRectanglarRegion(src_rect,tar_rect);
		if(common_rect.width <= 0 || common_rect.height <= 0 ) continue;

		if(isOverlayed(common_rect,ptar->second,psrc->second, type, &mask)){
			dst.push_back(*ptar_id);
		}
	}
	return dst;
}

size_t PatchLayer::getLayer(size_t i,bool from_top)const{
	size_t lorder = 0;

	if(from_top){
		std::list<size_t>::const_reverse_iterator pid = layer_order.rbegin();
		for(;lorder < layer_order.size()
				&& pid != layer_order.rend();lorder++, pid++){
			if(i == lorder){
				return *pid;
			}
		}
	}
	else{
		std::list<size_t>::const_iterator pid = layer_order.begin();
		for(;lorder < layer_order.size()
				&& pid != layer_order.end();lorder++, pid++){
			if(i == lorder){
				return *pid;
			}
		}
	}
	return UINT_MAX;
}

size_t PatchLayer::getOrder(size_t ID)const{
	std::list<size_t>::const_iterator pid;
	size_t i=0;
	for(pid = layer_order.begin();
			pid != layer_order.end();pid++,i++){
		if(*pid==ID) return i;
	}
	return UINT_MAX;
}

size_t PatchLayer::size()const{
	return layer_order.size();
}

bool PatchLayer::isOverlayed(
		const CvRect& common_rect,
		const Patch& p1,
		const Patch& p2,
		Patch::Type type,
		Image* mask){
	int max_x = common_rect.x + common_rect.width;
	int max_y = common_rect.y + common_rect.height;

	if(mask!=NULL){
		assert(mask->getWidth()==p2.getMask(Patch::original).getWidth());
		assert(mask->getHeight()==p2.getMask(Patch::original).getHeight());
	}

	bool flag = false;
	CvRect rect = p2.getRect(Patch::original);
	for(int y = common_rect.y; y < max_y; y++){
		for(int x = common_rect.x; x < max_x; x++){
			if( p1.maskValue(x,y,type)==0.0f
				|| p2.maskValue(x,y,type)==0.0f){
				continue;
			}

			if( mask==NULL){
				return true;
			}

			flag = true;
			((float*)(mask->getIplImage()->imageData + 
					mask->getIplImage()->widthStep * (y - rect.y)
					))[x - rect.x] = 0.0f;
		}
	}
	//if(mask!=NULL) return flag;
	//return false;

	return flag;
}

size_t PatchLayer::getTopLayer()const{
	return *(layer_order.rbegin());
}
