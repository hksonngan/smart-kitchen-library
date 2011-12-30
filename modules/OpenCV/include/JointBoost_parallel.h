/*!
 * @file FeatureClassifierJointBoost.h
 * @author ∂∂À‹∆ÿªÀ
 * @date Last Change:2011/Nov/24.
 */
#ifndef __FEATURECLASSIFIER_JOINT_BOOSTING_PARALLEL_H__
#define __FEATURECLASSIFIER_JOINT_BOOSTING_PARALLEL_H__
#include "Feature.h"
#include "FeatureClassifierDecisionStumpForJointBoost.h"
#include <tbb/blocked_range.h>
#include <map>
#include "WeakClassifierSet.h"

namespace mmpl{
	namespace image{
		class JointBoostWeightUpdate{
			private:
				FeatureClassifierDecisionStumpForJointBoost* classifier;
				std::vector<Feature>* training_set;
				MmplVector* weight;
				size_t class_num;
			public:
				JointBoostWeightUpdate(
						FeatureClassifierDecisionStumpForJointBoost* classifier,
						std::vector<Feature>* training_set,
						MmplVector* weight,
						size_t class_num);
				void operator()(const tbb::blocked_range<size_t>& range)const;
		};

		class JointBoostFastPredict{
			private:
				std::vector<MmplVector>* likelihoods;
				Feature* sample;
				const WeakClassifierSet<FeatureClassifierDecisionStumpForJointBoost>* weak_classifiers;
			public:
				JointBoostFastPredict(
						std::vector<MmplVector>* likelihoods,
						Feature* sample,
						const WeakClassifierSet<FeatureClassifierDecisionStumpForJointBoost>* weak_classifiers);
				void operator()(const tbb::blocked_range<size_t>& range)const;
		};

		class JointBoostParallelCountUpLikelihood{
			private:
				MmplVector* likelihood;
				std::vector<MmplVector>* likelihoods;
				const std::vector<double>* weight;
			public:
				JointBoostParallelCountUpLikelihood(
						MmplVector* likelihood,
						std::vector<MmplVector>* likelihoods,
						const std::vector<double>* weight);
				void operator()(const tbb::blocked_range<size_t>& range)const;
		};

	} // namespace mmpl::image;
} // namepsace mmpl;

#endif
