#include "TableObjectManager.h"

#ifdef DEBUG
#include <highgui.h>
#include "sklcvutils.h"
#endif

using namespace skl;

TableObjectManager::TableObjectManager(float __learning_rate):_learning_rate(__learning_rate){
#ifdef DEBUG
	cv::namedWindow("bgs_result",0);
	cv::namedWindow("region_label",0);
	cv::namedWindow("human",0);
#endif
}

TableObjectManager::~TableObjectManager(){
#ifdef DEBUG
	cv::destroyWindow("bgs_result");
	cv::destroyWindow("region_label");
	cv::destroyWindow("human");
#endif
}

void TableObjectManager::compute(
		const cv::Mat& src,
		cv::Mat& human,
		std::vector<size_t>& put_objects,
		std::vector<size_t>& taken_objects){
	cv::Mat bgs_image;
	bgs_algo->compute(src,bgs_image);
#ifdef DEBUG
	cv::imshow("bgs_result",bgs_image);
#endif

	cv::Mat labels;
	size_t region_num = rl_algo->compute(src,bgs_image,labels);

#ifdef DEBUG
	cv::imshow("region_label",visualizeRegionLabel(labels,region_num));
#endif

	std::list<size_t> human_region;
	assert(hd_algo!=NULL);
	cv::Mat human_small;
	human_region = hd_algo->compute(src,labels,human_small);
#ifdef DEBUG
	cv::imshow("human",human_small);
#endif

	cv::Mat object_cand_small;
	cv::Mat object_mask = 255 - human_small;
	size_t object_cand_num = srd_algo->compute(labels, object_mask, object_cand_small);

	// set scale of label resolution to src image resolution;
	assert(0 == src.rows % labels.rows);
	int scale = src.rows / labels.rows;

	cv::Mat object_cand;
	if(scale > 1){
		human = cv::Mat(src.size(),CV_8UC1);
		cv::resize(human_small,human,human.size(),cv::INTER_NEAREST);
		object_cand = cv::Mat(src.size(),object_cand_small.type());
		cv::resize(object_cand_small,object_cand,object_cand.size(),cv::INTER_NEAREST);
	}
	else{
		human = human_small;
		object_cand = object_cand_small;
	}

	patch_model->setObjectLabels(
			src,
			human,
			object_cand,
			object_cand_num,
			&put_objects,
			&taken_objects);

	patch_model->update(src, object_mask, _learning_rate);
}

