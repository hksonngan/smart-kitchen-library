#include "RegionLabelingSimple.h"
using namespace skl;
/** 
 * @brief $BF`NI@hC<Bg!"0fB<@h@8$N%W%m%0%i%`!J7cB.!K$rMxMQ$7$?%i%Y%j%s%0(B
 * $BGX7J?'$O(B(0,0,0)$B$G$J$$$H$$$1$J$$(B
 * */
size_t RegionLabelingSimple::compute(
		const cv::Mat& img,
		const cv::Mat& _mask,
		cv::Mat& labels){
	cv::Mat mask;
	assert(CV_8UC1 == _mask.type());
	if(mask.isContinuous()){
		mask = _mask;
	}
	else{
		mask = _mask.clone();
	}
	unsigned char* pmask = mask.ptr<unsigned char>(0);

	labels = cv::Mat::zeros(mask.rows,mask.cols,CV_16SC1);
	assert(labels.isContinuous());
	short* plabel = labels.ptr<short>(0);
	labeling.Exec(pmask,plabel,mask.cols,mask.rows,true,threshold);
//	std::cerr << labeling.GetNumOfResultRegions() << std::endl;
	return labeling.GetNumOfResultRegions();

}

