#include "FeatureClassifierDecisionStumpForJointBoost_forhighspeed.h"
#include <cassert>
using namespace mmpl;
#include <cmath>

/***** TrainSampleData *****/
FeatureClassifierDecisionStumpForJointBoostTrainSampleData::FeatureClassifierDecisionStumpForJointBoostTrainSampleData(){

}

/***** Intermidiate Data *****/
FeatureClassifierDecisionStumpForJointBoostIntermidiateData::FeatureClassifierDecisionStumpForJointBoostIntermidiateData(){

}

FeatureClassifierDecisionStumpForJointBoostIntermidiateData::FeatureClassifierDecisionStumpForJointBoostIntermidiateData(const FeatureClassifierDecisionStumpForJointBoostIntermidiateData& other){
	b = other.b;
	a_plus_b = other.a_plus_b;
}

void FeatureClassifierDecisionStumpForJointBoostIntermidiateData::clear(){
	b.clear();
	a_plus_b.clear();
}

void FeatureClassifierDecisionStumpForJointBoostIntermidiateData::resize(size_t feature_dim){
	b.resize(feature_dim);
	a_plus_b.resize(feature_dim);
}

void FeatureClassifierDecisionStumpForJointBoostIntermidiateData::resize(size_t feature_index, size_t cand_thresh_size){
	assert(feature_index < b.size());
	assert(feature_index < a_plus_b.size());
	b[feature_index].resize(cand_thresh_size);
	a_plus_b[feature_index].resize(cand_thresh_size);
}

/********* RegressionStumpNodeTrain **********/

RegressionStumpNodeTrain::RegressionStumpNodeTrain(
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
		std::vector<size_t>* argmin_ds):
class_num(class_num),
s(current_step),
i(thresh_index),
d(d),
hasSelected(hasSelected),
flag_combi(flag_combi),
weight_a_positive(weight_a_positive),
weight_a_negative(weight_a_negative),
weight_b_positive(weight_b_positive),
weight_b_negative(weight_b_negative),
parent_nodes(parent_nodes),
min_errors(min_errors),
argmin_thresh_indice(argmin_thresh_indice),
argmin_ds(argmin_ds){}

void RegressionStumpNodeTrain::operator()(const tbb::blocked_range<size_t>& range)const{
	for(size_t combi = range.begin();combi != range.end(); combi++){
		Bitflag p1 = flag_combi->at(combi).first;
		Bitflag p2 = flag_combi->at(combi).second;
		Bitflag flag = p1 | p2;
		if(hasSelected!=NULL &&
				(hasSelected->at(d)&flag).any()){
			continue;
		}
		double w_a_pos1 = 0;
		double w_a_neg1 = 0;
		double w_b_pos1 = 0;
		double w_b_neg1 = 0;
		double w_a_pos2 = 0;
		double w_a_neg2 = 0;
		double w_b_pos2 = 0;
		double w_b_neg2 = 0;
		for(size_t c=0;c<class_num;c++){
			if(p1[c]){
				w_a_pos1 += weight_a_positive->at(c);
				w_a_neg1 += weight_a_negative->at(c);
				w_b_pos1 += weight_b_positive->at(c);
				w_b_neg1 += weight_b_negative->at(c);
			}
			else if(p2[c]){
				w_a_pos2 += weight_a_positive->at(c);
				w_a_neg2 += weight_a_negative->at(c);
				w_b_pos2 += weight_b_positive->at(c);
				w_b_neg2 += weight_b_negative->at(c);
			}
		}
		std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>::const_iterator ipn1,ipn2;
		ipn1 = parent_nodes->find(p1);
		ipn2 = parent_nodes->find(p2);
		assert(parent_nodes->end() != ipn1);
		assert(parent_nodes->end() != ipn2);
		double b = ipn1->second.b[d][s] * (w_b_pos1 + w_b_neg1)
				 + ipn2->second.b[d][s] * (w_b_pos2 + w_b_neg2);
		b /= (w_b_pos1 + w_b_neg1 + w_b_pos2 + w_b_neg2);

		double a_plus_b = ipn1->second.a_plus_b[d][s] * (w_a_pos1 + w_a_neg1)
						+ ipn2->second.a_plus_b[d][s] * (w_a_pos2 + w_a_neg2);
		a_plus_b /= (w_a_pos1 + w_a_neg1 + w_a_pos2 + w_a_neg2);


		double error = std::pow( 1-a_plus_b,2.0) * (w_a_pos1 + w_a_pos2)
					+  std::pow(-1-a_plus_b,2.0) * (w_a_neg1 + w_a_neg2)
					+  std::pow( 1-b,       2.0) * (w_b_pos1 + w_b_pos2)
					+  std::pow(-1-b,       2.0) * (w_b_neg1 + w_b_neg2);
		if(error < min_errors->at(combi)){
			min_errors->at(combi) = error;
			argmin_thresh_indice->at(combi) = i;
			argmin_ds->at(combi) = d;
		}
	}
}

void RegressionStumpUpdateWeightSequence::operator()(const tbb::blocked_range<size_t>& range)const{
	for(size_t c = range.begin();c!= range.end();c++){
		if(c==class_id){
			weight_a_positive->at(c) -= weight[c];
			weight_b_positive->at(c) += weight[c];
		}
		else{
			weight_a_negative->at(c) -= weight[c];
			weight_b_negative->at(c) += weight[c];
		}
	}
}
