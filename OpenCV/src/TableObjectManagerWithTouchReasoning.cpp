﻿/*!
 * @file TableObjectManagerWithTouchReasoning.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Sep/27
 * @date Last Change: 2012/Sep/27.
 */
#include "TableObjectManagerWithTouchReasoning.h"
#include "sklcv.h"
using namespace skl;


/*!
 * @brief デフォルトコンストラクタ
 */
TableObjectManagerWithTouchReasoning::TableObjectManagerWithTouchReasoning(
		float __learning_rate,
		cv::Ptr<BackgroundSubtractAlgorithm> __bgs_algo,
		cv::Ptr<RegionLabelingAlgorithm> __rl_algo,
		cv::Ptr<HumanDetector> __hd_algo,
		cv::Ptr<RegionLabelingAlgorithm> __srd_algo,
		cv::Ptr<RegionLabelingAlgorithm> __trd_algo,
		cv::Ptr<PatchModel> __patch_model):
	TableObjectManager(__learning_rate,__bgs_algo,__rl_algo,__hd_algo,__srd_algo,__patch_model),_trd_algo(__trd_algo)
{

}

/*!
 * @brief デストラクタ
 */
TableObjectManagerWithTouchReasoning::~TableObjectManagerWithTouchReasoning(){

}

void TableObjectManagerWithTouchReasoning::compute(
		const cv::Mat& src,
		cv::Mat& human,
		std::vector<size_t>& put_objects,
		std::vector<size_t>& taken_objects){
#ifdef DEBUG_TABLE_OBJECT_MANAGER
	StopWatch timer;
#else
	cv::Mat bgs_image;
	cv::Mat labels;
	cv::Mat static_region_labels;
	cv::Mat no_touched_fg;
#endif

	bg_subtract(src,bgs_image);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "BGS   :" << timer.lap() << std::endl;
	region_num = 
#endif
		_rl_algo->compute(src,bgs_image,labels);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "LABEL :" <<timer.lap() << std::endl;
#endif

	std::list<size_t> human_region;
	assert(_hd_algo!=NULL);
	cv::Mat human_small = cv::Mat(labels.size(),CV_8UC1);
	human_region = _hd_algo->compute(src,labels,human_small);
#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "HUMAN :" << timer.lap() << std::endl;
#endif

	static_region_labels = cv::Mat(human_small.size(),CV_16SC1);
	size_t object_cand_num = _srd_algo->compute(labels,255-human_small, static_region_labels);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "STATIC:" << timer.lap() << std::endl;
#endif

	cv::Mat object_cand_small = cv::Mat(human_small.size(),CV_16SC1);
	//	cv::Mat object_cand_small = static_region_labels;
	object_cand_num = _trd_algo->compute(static_region_labels,human_small,object_cand_small);

	// mat type = CV_32FC1
	cv::Mat non_update_mask_small = getLabelDiff(labels,object_cand_small);
	// mat type = CV_32FC1
	cv::Mat no_touched_fg_small = getLabelDiff(static_region_labels,object_cand_small);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "TOUCH:" << timer.lap() << std::endl;
#endif

	// set scale of label resolution to src image resolution;
	assert(0 == src.rows % labels.rows);
	int scale = src.rows / labels.rows;

	cv::Mat object_cand,non_update_mask;
	if(scale > 1){
		if(human.size()!=src.size()){
			human = cv::Mat::zeros(src.size(),human_small.type());
		}
		skl::resize_label<unsigned char>(human_small,scale,human);
		object_cand = skl::resize_label<short>(object_cand_small,scale);
		non_update_mask = skl::resize_label<float>(non_update_mask_small,scale);
		//		no_touched_fg = skl::resize_label<float>(no_touched_fg_small,scale);
	}
	else{
		human = human_small;
		object_cand = object_cand_small;
		non_update_mask = non_update_mask_small;
		//		no_touched_fg = no_touched_fg_small;
	}

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "RESIZE:" << timer.lap() << std::endl;
#endif

	_patch_model->setObjectLabels(
			src,
			human,
			object_cand,
			object_cand_num,
			&put_objects,
			&taken_objects);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "PATCH :" << timer.lap() << std::endl;
#endif

	cv::Mat mask = _patch_model->updated_mask();

	//	no_touched_fg = Patch::blur_mask(no_touched_fg,PATCH_DILATE);

	non_update_mask = Patch::blur_mask(non_update_mask,PATCH_DILATE);
	non_update_mask = cv::max(_patch_model->updated_mask(),non_update_mask);
	//	non_update_mask = cv::max(no_touched_fg,non_update_mask);
	non_update_mask *= _learning_rate;
	non_update_mask += 1.0 - _learning_rate;


#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "MASK  :" << timer.lap() << std::endl;
#endif

	bg_update(src,non_update_mask);

#ifdef DEBUG_TABLE_OBJECT_MANAGER
	std::cerr << "SETBG :" << timer.stop() << std::endl;
#endif
}

