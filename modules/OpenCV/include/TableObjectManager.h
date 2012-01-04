#ifndef __SKL_TABLE_OBJECT_MANAGER_H__
#define __SKL_TABLE_OBJECT_MANAGER_H__

#include <list>
#include "BackgroundSubtractAlgorithm.h"
#include "PatchModel.h"
#include "FilterMat2Mat.h"

namespace skl{

	class TableObjectManager{
		public:
			typedef FilterMat2Mat<size_t> RegionLabelingAlgorithm;
			typedef FilterMat2Mat<std::list<size_t> > HumanDetector;
			TableObjectManager(float __learning_rate=0.05);
			~TableObjectManager();
			void setAlgorithms(
					BackgroundSubtractAlgorithm* bgs_algo,
					RegionLabelingAlgorithm* rl_algo,
					HumanDetector* human_detect_algo,
					RegionLabelingAlgorithm* static_region_detect_algo,
					PatchModel* patch_model
					){
				this->bgs_algo = bgs_algo;
				this->rl_algo = rl_algo;
				this->hd_algo = human_detect_algo;
				this->srd_algo = static_region_detect_algo;
				this->patch_model = patch_model;
			}

			void compute(const cv::Mat& src, cv::Mat& human, std::vector<size_t>& put_objects, std::vector<size_t>& taken_objects);


			float learning_rate()const{return _learning_rate;}
			void learning_rate(float __learning_rate){_learning_rate = __learning_rate;}
		protected:
			BackgroundSubtractAlgorithm* bgs_algo;
			RegionLabelingAlgorithm* rl_algo;
			HumanDetector* hd_algo;
			RegionLabelingAlgorithm* srd_algo;
			PatchModel* patch_model;
			float _learning_rate;

		private:
};

}

#endif // __TABLE_OBJECT_MANAGER_H__
