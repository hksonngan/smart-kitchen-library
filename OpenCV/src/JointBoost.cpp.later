/*!
 * @file FeatureClassifierJointBoost.cpp
 * @author 橋本敦史
 * @date Last Change:2011/Dec/05.
 */

#include "FeatureClassifierJointBoost.h"
#include "FeatureClassifierJointBoost_parallel.h"
#include <set>
#include <map>
#include <climits>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace mmpl;
using namespace mmpl::image;

/*!
 * @brief デフォルトコンストラクタ
 */
FeatureClassifierJointBoost::FeatureClassifierJointBoost(bool isApproximation):
	isApproximation(isApproximation),
	class_num(0),
	bg_class(UINT_MAX)
{
}

/*!
 * @brief デストラクタ
 */
FeatureClassifierJointBoost::~FeatureClassifierJointBoost(){}

/*!
 * @brief 弱識別器を追加する
 * @param weak_classifier 追加する弱識別器
 * @param weight 弱識別器の重み
 * */
void FeatureClassifierJointBoost::addWeakClassifier(const FeatureClassifierDecisionStumpForJointBoost& weak_classifier,double weight){
	param.push_back(weak_classifier);
	param.getConfidenceWeight()->push_back(weight);
}

/*!
 * @brief 弱識別器の数を取得する
 * */
size_t FeatureClassifierJointBoost::getWeakClassifierNum()const{
	return param.size();
}

bool FeatureClassifierJointBoost::train(const std::vector<Feature>& _training_set,const MmplVector* _weight){
	if(_training_set.empty()){ return false; }
	if(_training_set[0].size()==0){ return false; }
	//std::cerr << __FILE__ << ": " << __LINE__ << std::endl;

	// BoostingJointから移植(特徴はコピーせず、setSubsetBitflagを利用)
	std::vector<FeatureClassifierDecisionStumpForJointBoost> temp_classifiers();
	std::vector<Feature> training_set(_training_set);

	if(class_num==0){
		// クラス数を取得する
		for(size_t i=0;i<training_set.size();i++){
			unsigned int teacher_class_id = training_set[i].getTeacherClassID();
			if(teacher_class_id > class_num){
				class_num = teacher_class_id;
			}
		}
		// teacher class idは0始まりなので、クラス数は[最大id+1]となる
		class_num++;
	}
	//	std::cerr << __FILE__ << "Notice: class_num = " << class_num << std::endl;

	MmplVector weight;
	if(_weight!=NULL){
		assert(static_cast<unsigned int>(_weight->size()) == training_set.size() * class_num);
		weight = *_weight;
		weight_normalizing_factor = weight.getSum();
	}
	else{
		setWeight(training_set,&weight);
		weight_normalizing_factor = training_set.size() * class_num;
//		std::cerr << weight_normalizing_factor << std::endl;
//		std::cerr << weight.getSum() << std::endl;
	}

	if(isApproximation){
		return train_approx(training_set,&weight,class_num);
	}

	if(class_num < MAX_CLASS_WITH_PERFECT_JOINT_BOOST){
		std::cerr << "WARNING: too many classes to train 2^N-1 weak classifier." << std::endl;
		return false;
	}
	return train_perfect(training_set,&weight,class_num);
}

bool FeatureClassifierJointBoost::train_perfect(std::vector<Feature>& training_set,MmplVector* weight,unsigned int class_num){
	// 2^C-1個の弱識別器を全て作成する

	// trainingSetの各次元毎のsorting結果を共有するための変数
	FeatureClassifierDecisionStumpForJointBoostTrainSampleData train_set_data;
	bool hasSorted = false;

	param.resize(std::pow(2.0,(int)class_num)-1);
	param.getConfidenceWeight()->resize(std::pow(2.0,(int)class_num)-1);

	double target_class_weight_sum;
	for(unsigned long long bitflag=1;
			bitflag<=static_cast<unsigned long>(param.size());bitflag++){
		param[bitflag-1].getParam()->setSubsetBitflag(Bitflag(bitflag));
		param[bitflag-1].getParam()->setClassNum(class_num);
		param[bitflag-1].train(training_set,weight,&train_set_data,&target_class_weight_sum);
		// weightを更新する
		updateWeight(training_set,weight,&param[bitflag-1]);
		param.getConfidenceWeight()->at(bitflag-1) = target_class_weight_sum;
	}

	return true;
}


bool FeatureClassifierJointBoost::train_approx(std::vector<Feature>& training_set,MmplVector* weight,unsigned int class_num){
	// 有効な弱識別器をBeam Search(記憶するノードは1つだけ)によりGreedyに選出する
	assert(class_num <= JOINT_BOOST_MAX_CLASS_NUM);

	// trainingSetの各次元毎のsorting結果を共有するための変数
	FeatureClassifierDecisionStumpForJointBoostTrainSampleData train_set_data;

	std::cerr << "===== Training Start =====" << std::endl;
	FeatureClassifierDecisionStumpForJointBoost candidate;

	// 学習用セットに固有の情報を抽出
	candidate.getParam()->setClassNum(class_num);
	candidate.getTrainSampleData(training_set,*weight,&train_set_data);

	double prev_total_weight = DBL_MAX;
	size_t feature_dim = training_set[0].size();
	size_t max_iter = 8;
	double satisfying_weight = 0.3;
	double weak_classifier_weight;

	std::vector<Bitflag> hasSelected(feature_dim,0),hasSelected_set(feature_dim,0);

	for(size_t boosting_iter = 0; boosting_iter < max_iter; boosting_iter++){

		// 単独クラスのみの識別器を学習する
		std::vector<Bitflag> flag_vec(class_num);
		std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer> intermidiate_data_map;
		MmplVector temp_weight = *weight;
		bool isUpdated = false;
		for(size_t c=0; c < flag_vec.size(); c++){
			if(c == bg_class) continue;
			flag_vec[c][c] = true;
			FeatureClassifierDecisionStumpForJointBoost candidate;
			FeatureClassifierDecisionStumpForJointBoostIntermidiateData intermidiate_data;
			candidate.getParam()->setSubsetBitflag(flag_vec[c]);
			candidate.getParam()->setClassNum(class_num);
			candidate.train(training_set,weight,&train_set_data,&weak_classifier_weight,&intermidiate_data,&hasSelected);

			if(candidate.getError()==DBL_MAX){
				// all feature dim has been selected.
				continue;
			}
			// weightを更新する
			temp_weight = *weight;
			double current_total_weight = updateWeight(training_set,weight,&candidate);
			if(prev_total_weight < current_total_weight){
				// エラーが増えてしまったので元に戻す
				*weight = temp_weight;
				continue;
			}
			isUpdated = true;
			//double weighted_error_rate = candidate.getError() / (training_set.size()*class_num);
//			double alpha = weighted_error_rate * log((4-weighted_error_rate)/weighted_error_rate)/2;
//			std::cerr << "alpha: " << alpha << std::endl;
			std::cerr << "classifier_confidence: " << weak_classifier_weight << std::endl;
//			addWeakClassifier(candidate,weak_classifier_weight);
			addWeakClassifier(candidate,prev_total_weight);
			intermidiate_data_map[flag_vec[c]] = intermidiate_data;
			prev_total_weight = current_total_weight;
//			hasSelected[candidate.getParam()->getFocusingFeature()] |= candidate.getParam()->getSubsetBitflag();
		}

		std::cerr << "Selected Feature Matrix" << std::endl;
/*		for(size_t d=0; d< feature_dim; d++){
			std::cerr << hasSelected[d] << std::endl;
		}
*/
		std::cerr << "-----------------------" << std::endl;

		if(prev_total_weight < satisfying_weight) continue;
		if(!isUpdated) break;

		//	double standard_weight_sum = weight->size()*1000;
		while(intermidiate_data_map.size()>2){
			std::cerr << "Now training " << param.size() << "th weak classifier" << std::endl;

			FeatureClassifierDecisionStumpForJointBoostIntermidiateData intermidiate_data;
			candidate.getParam()->setClassNum(class_num);
			candidate.train(
					training_set,
					weight,
					train_set_data,
					&weak_classifier_weight,
					&intermidiate_data_map,
					&intermidiate_data,
					&hasSelected_set
					);
			if(candidate.getError()==DBL_MAX){
				// no more available feature set. All dim are selected.
				break;
			}

			// weightを更新する
			temp_weight = *weight;
			double total_weight = updateWeight(training_set,weight,&candidate);

			if(prev_total_weight <= total_weight){
				// 改善が見られないので、追加せずに, この回のイテレーションを止める
				*weight = temp_weight;
				break;
			}

			// 改善が見られるので、新しく学習したものをモデルに追加する
//			std::cerr << "alpha: " << alpha << std::endl;
			std::cerr << "classifier_confidence: " << weak_classifier_weight << std::endl;
//			addWeakClassifier(candidate,weak_classifier_weight);
			addWeakClassifier(candidate,prev_total_weight);
			intermidiate_data_map[candidate.getParam()->getSubsetBitflag()] = intermidiate_data;
			prev_total_weight = total_weight;
//			hasSelected_set[candidate.getParam()->getFocusingFeature()] |= candidate.getParam()->getSubsetBitflag();
		}
	}

	std::cerr << "final total weight: " << prev_total_weight << std::endl;
	std::cerr << "param size: " << param.size() << std::endl;
	std::cerr << "confidence: " << param.getConfidenceWeight()->size() << std::endl;
	return true;
}

void FeatureClassifierJointBoost::setWeight(const std::vector<Feature>& training_set,MmplVector* weight)const{
	weight->assign(training_set.size() * class_num,0.0);

	std::vector<size_t> cnt_sample_num(class_num,0);
	for(size_t i=0;i<training_set.size();i++){
		int class_id = training_set[i].getTeacherSignal();
		cnt_sample_num[class_id]++;
	}
	std::vector<double> pos_weight(class_num,0.0);
	std::vector<std::vector<double> > neg_weight(class_num,pos_weight);

	for(size_t c=0;c<class_num;c++){
		pos_weight[c] = (double)training_set.size() / ( 2 * cnt_sample_num[c] );
		double neg_weight_base = (double)training_set.size() / (2 * (class_num-1));
//		std::cerr << "class " << c << "'s sample number  : " << cnt_sample_num[c] << std::endl;
//		std::cerr << "class " << c << "'s positive weight: " << pos_weight[c] << std::endl << "negative weights: ";

		for(size_t n = 0; n < class_num; n++){
			if(c==n) continue;
			neg_weight[c][n] = neg_weight_base / cnt_sample_num[n];
//			std::cerr << n << "= " << neg_weight[c][n] << " ";
		}
//		std::cerr << std::endl;
	}

	std::vector<double> pos_weight_sum(class_num,0.0);
	std::vector<double> neg_weight_sum(class_num,0.0);
	for(size_t i=0,idx=0;i<training_set.size();i++){
		size_t c = training_set[i].getTeacherSignal();
		for(size_t n=0;n<class_num;n++,idx++){
			if(c==n){
				weight->at(idx) = pos_weight[c];
				pos_weight_sum[c] += pos_weight[c];
			}
			else{
				weight->at(idx) = neg_weight[n][c];
				neg_weight_sum[c] += neg_weight[n][c];
			}
		}
	}

/*
	std::cerr << ( (double)training_set.size() / 2 ) << std::endl;
	for(size_t c = 0; c < class_num; c++){
		std::cerr << "class " << c << std::endl;
		std::cerr << "pos_weight_sum = " << pos_weight_sum[c] << ", neg_weight_sum = " << neg_weight_sum[c] << std::endl;
	}
*/
}

double FeatureClassifierJointBoost::updateWeight(std::vector<Feature>& training_set, MmplVector* weight, FeatureClassifierDecisionStumpForJointBoost* classifier)const{
	const MmplVector* likelihood;
	double weight_sum = 0.0;
	
	tbb::task_scheduler_init TbbInit;
	tbb::parallel_for(
				tbb::blocked_range<size_t>(0, training_set.size()),
				JointBoostWeightUpdate(
					classifier,
					&training_set,
					weight,
					class_num),
				tbb::auto_partitioner()
			);
	TbbInit.terminate();


	for(size_t i=0,idx=0;i<training_set.size();i++){
		for(unsigned int c = 0;c < class_num;c++,idx++){
			weight_sum += weight->at(idx);
		}
	}
	weight_sum /= weight_normalizing_factor;
	std::cerr << "weight_sum = " << weight_sum << std::endl;
	return weight_sum;
}

double FeatureClassifierJointBoost::predict(Feature* testSample)const{
	return predict(testSample,0.0);
}

double FeatureClassifierJointBoost::predict(Feature* testSample, double min_weight)const{
	// 計算結果の格納先
	assert(param.size()>0);
	size_t class_num = param[0].getParam()->getClassNum();

	size_t active_weak_classifier_size = param.size();
	std::vector<double>::const_iterator icw = param.getConfidenceWeight()->begin();

	// 使用する弱識別器の上限を決める、重みの和を得る
//	std::vector<double> weight_sum(class_num,0.0);
	for(active_weak_classifier_size=0;
			active_weak_classifier_size < param.size();
			active_weak_classifier_size++, icw++){
/*		for(size_t c=0;c<class_num;c++){
			if(param[active_weak_classifier_size].getParam()->getSubsetBitflag()[c]){
				weight_sum[c] += *icw;
			}
		}
*/		if(*icw < min_weight){
			break;
		}
	}

	// 一時的な計算結果の格納先
	std::vector<MmplVector> likelihoods(active_weak_classifier_size);


	tbb::task_scheduler_init TbbInit;
	tbb::parallel_for(
			tbb::blocked_range<size_t>(0,active_weak_classifier_size),
			JointBoostFastPredict(
				&likelihoods,testSample,&param),
			tbb::auto_partitioner());
	TbbInit.terminate();
/*	std::cerr << "=====================" << std::endl;
	for(size_t i=0;i<active_weak_classifier_size;i++){
		std::cerr << likelihoods[i] << std::endl;
	}
*/
	MmplVector temp;
	temp.resize(class_num,0.0);
	TbbInit.initialize();
	tbb::parallel_for(
			tbb::blocked_range<size_t>(0,class_num),
			JointBoostParallelCountUpLikelihood(
				&temp,&likelihoods,param.getConfidenceWeight()),
			tbb::auto_partitioner());
	TbbInit.terminate();

	for(size_t c = 0; c < class_num; c++){
		temp[c] /= active_weak_classifier_size;
/*		if(weight_sum[c]==0){
			temp[c] = 0;
		}
		else{
			temp[c] /= weight_sum[c];
		}
*/	}
/*	std::cerr << "---------------------" << std::endl;
	std::cerr << temp << std::endl;
*/
	// 最終的な出力を返す
	testSample->setLikelihood(temp);
	return temp.max();
}

void FeatureClassifierJointBoost::setApproximationFlag(bool isApproximation){
	this->isApproximation = isApproximation;
}

void FeatureClassifierJointBoost::setBackgroundClass(size_t bg_class){
	this->bg_class = bg_class;
}
