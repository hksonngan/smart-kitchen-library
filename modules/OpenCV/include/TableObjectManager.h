#ifndef __SKL_TABLE_OBJECT_MANAGER_H__
#define __SKL_TABLE_OBJECT_MANAGER_H__

#include <list>
#include "BackgroundSubtractAlgorithm.h"
#include "FilterMat2Mat.h"

namespace skl{

	class TableObjectManager{
		public:
			typedef FilterMat2Mat<size_t> RegionLabelingAlgorithm;
			typedef FilterMat2Mat<std::list<size_t> > HumanDetector;
			TableObjectManager();
			~TableObjectManager();
			void setAlgorithms(
					BackgroundSubtractAlgorithm* bgs_algo,
					RegionLabelingAlgorithm* rl_algo,
					HumanDetector* human_detect_algo,
					RegionLabelingAlgorithm* static_region_detect_algo
					){
				this->bgs_algo = bgs_algo;
				this->rl_algo = rl_algo;
				this->hd_algo = human_detect_algo;
				this->srd_algo = static_region_detect_algo;
			}

			void compute(const cv::Mat& src,cv::Mat& human, std::list<size_t>& put_objects, std::list<size_t>& taken_objects);
		protected:
			BackgroundSubtractAlgorithm* bgs_algo;
			RegionLabelingAlgorithm* rl_algo;
			HumanDetector* hd_algo;
			RegionLabelingAlgorithm* srd_algo;
		private:
	};

}

#endif // __TABLE_OBJECT_MANAGER_H__
