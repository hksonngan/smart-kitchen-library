/*!
 * @file RegressionStump.h
 * @author 橋本敦史
 * @date Last Change:2011/Nov/27.
 */
#ifndef __SKL_REGRESSION_STUMP_H__
#define __SKL_REGRESSION_STUMP_H__

#include <cv.h>
//#include "RegressionStump_parallel.h"

namespace skl{

	/*!
	 * @class 繰り返し同じ学習セットを適用する際に二回目以降のソーティング作業を省略するためのクラス
	 */
	class RegressionStumpTrainDataIndex{
		
	}

/*!
 * @class JointBoosting等で用いられる決定木のノードの帰り値を回帰としたモデル
 */
class RegressionStump:public CvStatModel{
	public:
		RegressionStump();
		RegressionStump(const RegressionStump& other);
		virtual ~RegressionStump();

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

		// オリジナル
		bool train(

				const std::vector<Feature>& ,
				const MmplVector* weight=NULL);

		bool train(
				const std::vector<Feature>& ,
				const MmplVector* weight,
				RegressionStumpTrainSampleData* train_set_data,
				double* target_class_weight_sum,
				RegressionStumpIntermidiateData* current_node=NULL,
				std::vector<Bitflag>* hasSelected=NULL);

		bool train(
				const std::vector<Feature>& train_set,
				const MmplVector* weight,
				const RegressionStumpTrainSampleData& train_set_data,
				double* target_class_weight_sum,
				std::map<Bitflag, RegressionStumpIntermidiateData,BitflagComparer>* parent_nodes,
				RegressionStumpIntermidiateData* intermidiate_data,
				std::vector<Bitflag>* hasSelected=NULL);

		void getTrainSampleData(
				const std::vector<Feature>& trainSet,
				const MmplVector& weight,
				RegressionStumpTrainSampleData* train_set_data)const;


		RegressionStumpTrainSampleData getTrainSampleData(const std::vector<Feature>&);
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
				RegressionStumpIntermidiateData* node = NULL
				)const;
		double _error;
	private:
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

		void invertSortingMap(
				const std::vector<std::vector<size_t> >& SortingMap,
				std::vector<std::vector<size_t> >* SortingMap_inv)const;
};


} // namespace skl
#endif // __SKL_REGRESSION_STUMP_H__
