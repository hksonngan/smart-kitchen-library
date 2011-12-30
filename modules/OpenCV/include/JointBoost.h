/*!
 * @file FeatureClassifierJointBoost.h
 * @author 橋本敦史
 * @date Last Change:2011/Dec/05.
 */
#ifndef __FEATURECLASSIFIER_JOINT_BOOSTING_H__
#define __FEATURECLASSIFIER_JOINT_BOOSTING_H__


#include "FeatureClassifierDecisionStumpForJointBoost.h"
#include "WeakClassifierSet.h"

#define MAX_CLASS_WITH_PERFECT_JOINT_BOOST 16

namespace mmpl{
	namespace image{
/*!
 * @class JointBoostによる識別器
 */
class FeatureClassifierJointBoost:
	public FeatureClassifierWithParam<WeakClassifierSet<FeatureClassifierDecisionStumpForJointBoost> >
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

	} // namespace mmpl::image;
} // namespace mmpl;

#endif // __FEATURECLASSIFIER_JOINT_BOOSTING_H__
