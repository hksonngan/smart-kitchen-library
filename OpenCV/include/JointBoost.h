/*!
 * @file JointBoost.h
 * @author 橋本敦史
 * @date Last Change:2011/Dec/05.
 */
#ifndef __SKL_JOINT_BOOSTING_H__
#define __SKL_JOINT_BOOSTING_H__

#include <vector.h>
#include <cv.h>
#include "RegressionStump.h"

#define MAX_CLASS_WITH_PERFECT_JOINT_BOOST 16

namespace skl{
	/*!
	 * @brief JointBoostによる識別器
	 */
	class JointBoost:public CvStatModel
	{
		public:
			FeatureClassifierJointBoost(bool isApproximation=true);
			virtual ~FeatureClassifierJointBoost();
			bool train(const std::vector<Feature>& ,const MmplVector* weight=NULL);
			double predict(Feature* testSample)const;
			double predict(Feature* testSample,double min_weight)const;
			size_t getWeakClassifierNum()const;
			void setApproximationFlag(bool isApproximation);
			void setBackgroundClass(size_t bg_class);

			// pure virtual functions from CvStatModel
			void save(const char* filename, const char* name=0);
			void load(const char* filename, const char* name=0);
			void write(CvFileStorage* storage, const char* name);
			void read(CvFileStorage* storage, CvFileNode* node);
			void clear();
			void setWeight(cv::Mat& weight);
			// this class support only CV_ROW_SAMPLE.
			bool train(
				const cv::Mat& train_data, // 1 row 1 sample. 
				const cv::Mat& responses, // teacher signal
				const cv::Mat& var_idx, // feature mask
				const cv::Mat& sample_idx); // sample mask
			float predict(const cv::Mat& sample);

	
		protected:
			bool isApproximation;
			size_t class_num;
			void addWeakClassifier(const FeatureClassifierDecisionStumpForJointBoost& weak_classifier,double weight=1);
			size_t bg_class;
			double weight_normalizing_factor;
			void setWeight(const std::vector<Feature>& training_set,MmplVector* weight)const;				

		private:
			bool train_perfect(std::vector<Feature>& ,MmplVector* weight,unsigned int class_num);
			bool train_approx(std::vector<Feature>& ,MmplVector* weight,unsigned int class_num);
			double updateWeight(
					std::vector<Feature>& training_set,
					MmplVector* weight,
					FeatureClassifierDecisionStumpForJointBoost* classifier)const;
/*			double evalErrorRate(
					std::vector<Feature>& training_set,
					MmplVector* weight,
					FeatureClassifierDecisionStumpForJointBoost* classifier)const;
*/
	};

} // namespace SKL;

#endif // __SKL_JOINT_BOOSTING_H__
