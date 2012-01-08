#include "FeatureClassifierJointBoost_parallel.h"

namespace mmpl{
	namespace image{

JointBoostWeightUpdate::JointBoostWeightUpdate(
		FeatureClassifierDecisionStumpForJointBoost* classifier,
		std::vector<Feature>* training_set,
		MmplVector* weight,
		size_t class_num){
	this->classifier = classifier;
	this->training_set = training_set;
	this->weight = weight;
	this->class_num = class_num;
}

void JointBoostWeightUpdate::operator()(
			const tbb::blocked_range<size_t>& range
		)const{
	for(size_t i = range.begin(); i != range.end(); i++){
		size_t idx = i * class_num;
		classifier->predict(&(training_set->at(i)),true);
		const MmplVector* likelihood = training_set->at(i).getLikelihood();

		for(unsigned int c = 0;c < class_num;c++){
			double z = (c==training_set->at(i).getTeacherClassID()) ? 1.0 : -1.0;
			if(weight->at(idx + c) == 0 ) continue;

			weight->at(idx + c) *= exp(-z*likelihood->at(c));

		}
	}

}

JointBoostFastPredict::JointBoostFastPredict(
						std::vector<MmplVector>* likelihoods,
						Feature* sample,
						const WeakClassifierSet<FeatureClassifierDecisionStumpForJointBoost>* weak_classifiers):
likelihoods(likelihoods),sample(sample),weak_classifiers(weak_classifiers){
}


void JointBoostFastPredict::operator()(const tbb::blocked_range<size_t>& range)const{
	for(size_t i = range.begin(); i != range.end(); i++){
		Feature _sample(*sample);
		weak_classifiers->at(i).predict(&_sample);
		likelihoods->at(i) = *(_sample.getLikelihood());
	}
}

/************** CountUpLikelihood ****************/
JointBoostParallelCountUpLikelihood::JointBoostParallelCountUpLikelihood(
		MmplVector* likelihood,
		std::vector<MmplVector>* likelihoods,
		const std::vector<double>* weight
		)
	:likelihood(likelihood),likelihoods(likelihoods),weight(weight){
}

void JointBoostParallelCountUpLikelihood::operator()(const tbb::blocked_range<size_t>& range)const{
	for(size_t c = range.begin();c != range.end(); c++){
		for(size_t i = 0; i < likelihoods->size(); i++){
			// 重みづき和にする
//			likelihood->at(c) += likelihoods->at(i)[c] * weight->at(i);
			likelihood->at(c) += likelihoods->at(i)[c];
		}
	}
}


	} // namespace mmpl::image;
} // namepsace mmpl;

