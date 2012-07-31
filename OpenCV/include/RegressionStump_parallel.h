/*!
 * @file FeatureClassifierDecisionStumpForJointBoost.h
 * @author a_hasimoto
 * @date Last Change:2011/Nov/27.
 */
#ifndef __FEATURE_CLASSIFIER_DECISION_STAMP_FOR_JOINT_BOOST_FOR_HIGH_SPEED_H__
#define __FEATURE_CLASSIFIER_DECISION_STAMP_FOR_JOINT_BOOST_FOR_HIGH_SPEED_H__

#ifndef JOINT_BOOST_MAX_CLASS_NUM
#define JOINT_BOOST_MAX_CLASS_NUM 128
#endif

#include <bitset>
#include <vector>
#include <map>
#include <tbb/blocked_range.h>

namespace mmpl{
typedef std::bitset<JOINT_BOOST_MAX_CLASS_NUM> Bitflag;
class BitflagComparer{
	public:
		bool operator()(
				const Bitflag& a,
				const Bitflag& b)const{
			for(size_t i=JOINT_BOOST_MAX_CLASS_NUM-1;i<JOINT_BOOST_MAX_CLASS_NUM;i--){
				if(a[i] && !b[i]){
					return false;
				}
				else if(!a[i] && b[i]){
					return true;
				}
			}
			return false;
		}
};

class FeatureClassifierDecisionStumpForJointBoostTrainSampleData{
	public:
		FeatureClassifierDecisionStumpForJointBoostTrainSampleData();
		size_t feature_dim;
		std::vector<std::vector<size_t> > sorting_map;
		std::vector<std::vector<size_t> > sorting_map_inv;
		std::vector<std::vector<double> > thresh_candidates;
		std::vector<double> default_total_weight;
};

class FeatureClassifierDecisionStumpForJointBoostIntermidiateData{
	public:
		FeatureClassifierDecisionStumpForJointBoostIntermidiateData();
		FeatureClassifierDecisionStumpForJointBoostIntermidiateData(const FeatureClassifierDecisionStumpForJointBoostIntermidiateData& other);
		void clear();
		void resize(size_t feature_dim);
		void resize(size_t feature_index, size_t cand_thresh_size);
		std::vector<std::vector<double> > b;
		std::vector<std::vector<double> > a_plus_b;
};

class RegressionStumpNodeTrain{
	private:
		size_t class_num;
		size_t s;// current step
		size_t i;
		size_t d;// current feature_index
		const std::vector<Bitflag>* hasSelected;
		const std::vector<std::pair<Bitflag,Bitflag> >* flag_combi;
		const std::vector<double>* weight_a_positive;
		const std::vector<double>* weight_a_negative;
		const std::vector<double>* weight_b_positive;
		const std::vector<double>* weight_b_negative;
		const std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>* parent_nodes;
		std::vector<double>* min_errors;
		std::vector<size_t>* argmin_thresh_indice;
		std::vector<size_t>* argmin_ds;
	public:
		RegressionStumpNodeTrain(
				size_t class_num,
				size_t current_step,
				size_t thresh_index,
				size_t d,
				const std::vector<Bitflag>* hasSelected,
				const std::vector<std::pair<Bitflag,Bitflag> >* flag_combi,
				const std::vector<double>* weight_a_positive,
				const std::vector<double>* weight_a_negative,
				const std::vector<double>* weight_b_positive,
				const std::vector<double>* weight_b_negative,
				const std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>* parent_nodes,
				std::vector<double>* min_errors,
				std::vector<size_t>* argmin_thresh_indice,
				std::vector<size_t>* argmin_ds
				);
		void operator()(const tbb::blocked_range<size_t>& range)const;
};

class RegressionStumpUpdateWeightSequence{
	private:
		size_t class_id;
		const double* weight;
		std::vector<double>* weight_a_positive;
		std::vector<double>* weight_a_negative;
		std::vector<double>* weight_b_positive;
		std::vector<double>* weight_b_negative;
	public:
		RegressionStumpUpdateWeightSequence(
				size_t class_id,
				const double* weight,
				std::vector<double>* weight_a_positive,
				std::vector<double>* weight_a_negative,
				std::vector<double>* weight_b_positive,
				std::vector<double>* weight_b_negative):
			class_id(class_id),weight(weight),
			weight_a_positive(weight_a_positive),
			weight_a_negative(weight_a_negative),
			weight_b_positive(weight_b_positive),
			weight_b_negative(weight_b_negative)
		{
		}
		void operator()(const tbb::blocked_range<size_t>& range)const;

};

} // namespace mmpl
#endif
