/*!
 * @file FeatureClassifierDecisionStumpForJointBoost.h
 * @author 橋本敦史
 * @date Last Change:2011/Nov/27.
 */
#ifndef __FEATURE_CLASSIFIER_DECISION_STAMP_FOR_JOINT_BOOST_H__
#define __FEATURE_CLASSIFIER_DECISION_STAMP_FOR_JOINT_BOOST_H__

#include "FeatureClassifierWithParam.h"
#include "FeatureClassifierDecisionStumpForJointBoost_forhighspeed.h"
#include "Printable.h"

namespace mmpl{


/*!
 * @class 決定木のノードの値([1]A. Torralba, K.P.Murphy and W.T.Freeman,”Sharing features: efficient boosting procedures for multiclass object detection”に準拠)
 */
class FeatureClassifierDecisionStumpForJointBoostParams:public Printable<FeatureClassifierDecisionStumpForJointBoostParams>{
	public:
		FeatureClassifierDecisionStumpForJointBoostParams();
		virtual ~FeatureClassifierDecisionStumpForJointBoostParams();
		std::string print()const;
		void scan(const std::string& content);

		/* Accessor */
		void setThreshold(double threshold);
		double getThreshold()const;
		void setA(double a);
		double getA()const;
		void setB(double b);
		double getB()const;
		void setK(const std::vector<double>& k);
		double getK(size_t idx)const;
		unsigned int getClassNum()const;
		void setClassNum(unsigned int class_num);
		size_t getFocusingFeature()const;
		void setFocusingFeature(size_t focusing_feature);

		FeatureClassifierDecisionStumpForJointBoostParams& operator=(const FeatureClassifierDecisionStumpForJointBoostParams& other);

		const Bitflag& getSubsetBitflag()const;
		void setSubsetBitflag(const Bitflag& bitflag);


	protected:
		// 閾値
		double threshold;

		// JointBoostの論文[1]に準拠したパラメタ群
		double a;
		double b;
		std::vector<double> k;
		// その他、弱識別器を特徴づけるパラメタ
		Bitflag subset_bitflag;
		unsigned int class_num;
		size_t focusing_feature;
	private:
};


/*!
 * @class 決定木のノードを表す。AdaBoostと組み合わせて特徴量選出などに用いられる
 */
class FeatureClassifierDecisionStumpForJointBoost:public FeatureClassifierWithParam<FeatureClassifierDecisionStumpForJointBoostParams>{
	public:
		FeatureClassifierDecisionStumpForJointBoost();
		FeatureClassifierDecisionStumpForJointBoost(const FeatureClassifierDecisionStumpForJointBoost& other);
		virtual ~FeatureClassifierDecisionStumpForJointBoost();
		bool train(
				const std::vector<Feature>& ,
				const MmplVector* weight=NULL);

		bool train(
				const std::vector<Feature>& ,
				const MmplVector* weight,
				FeatureClassifierDecisionStumpForJointBoostTrainSampleData* train_set_data,
				double* target_class_weight_sum,
				FeatureClassifierDecisionStumpForJointBoostIntermidiateData* current_node=NULL,
				std::vector<Bitflag>* hasSelected=NULL);

		bool train(
				const std::vector<Feature>& train_set,
				const MmplVector* weight,
				const FeatureClassifierDecisionStumpForJointBoostTrainSampleData& train_set_data,
				double* target_class_weight_sum,
				std::map<Bitflag, FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>* parent_nodes,
				FeatureClassifierDecisionStumpForJointBoostIntermidiateData* intermidiate_data,
				std::vector<Bitflag>* hasSelected=NULL);

		void getTrainSampleData(
				const std::vector<Feature>& trainSet,
				const MmplVector& weight,
				FeatureClassifierDecisionStumpForJointBoostTrainSampleData* train_set_data)const;


		FeatureClassifierDecisionStumpForJointBoostTrainSampleData getTrainSampleData(const std::vector<Feature>&);
		double predict(Feature* testSample)const;
		double predict(Feature* testSample,bool addBandK)const;

		double getError()const;
	protected:
		void calcRegressionParams(const std::vector<Feature>& trainSet,const MmplVector* weight);
		void calcRegressionParamsParallel(
				const std::vector<Feature>& trainSet,
				const MmplVector* weight,
				const std::vector<std::pair<Bitflag,Bitflag> >& flag_combi,
				std::vector<double>* errors);

		bool calcSortingMap(
				const std::vector<Feature>& trainSet,
				std::vector<std::vector<size_t> >* SortingMap,
				std::vector<std::vector<double> >* thresh_candidates)const;
		bool getThresholdAndMinError(
				const std::vector<Feature>& trainSet,
				const std::vector<size_t>& SortingMap,
				size_t d,
				const std::vector<double>& weight,
				const Bitflag& bitflag,
				double* min_error,
				double* threshold,
				double* a,
				double* b,
				double* target_class_weight_sum,
				FeatureClassifierDecisionStumpForJointBoostIntermidiateData* node = NULL
				)const;
		double _error;
	private:
		void invertSortingMap(
				const std::vector<std::vector<size_t> >& SortingMap,
				std::vector<std::vector<size_t> >* SortingMap_inv)const;
};


} // namespace mmpl
#endif // __FEATURECLASSIFIERDECISIONSTAMP_H__

