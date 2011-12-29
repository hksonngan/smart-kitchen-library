#include "TableObjectManager.h"

#ifdef DEBUG
#include <highgui.h>
#include "RegionLabelingVisualize.h"
#endif

using namespace skl;

TableObjectManager::TableObjectManager(){
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
		std::list<size_t>& put_objects,
		std::list<size_t>& taken_objects){
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
	human_region = hd_algo->compute(src,labels,human);
#ifdef DEBUG
	cv::imshow("human",human);
#endif

	cv::Mat object_cand;
	size_t object_cand_num = srd_algo->compute(labels,255 - human,object_cand);

}
