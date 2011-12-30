/*!
 * @file FeatureClassifierDecisionStumpForJointBoost.cpp
 * @author 橋本敦史
 * @date Last Change:2011/Dec/05.
 */
#include "FeatureClassifierDecisionStumpForJointBoost.h"
#include "StringSplitter.h"
#include <sstream>

#include <set>
#include <climits>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace std;
using namespace mmpl;

namespace mmpl{


/**** Stat ****/
/*!
 * @brief デフォルトコンストラクタ
 */
FeatureClassifierDecisionStumpForJointBoostParams::FeatureClassifierDecisionStumpForJointBoostParams():
	threshold(0.0),
	a(0.0),
	b(0.0),
	subset_bitflag(0),
	class_num(0),
	focusing_feature(0){
}


/*!
 * @brief デストラクタ
 */
FeatureClassifierDecisionStumpForJointBoostParams::~FeatureClassifierDecisionStumpForJointBoostParams(){

}

/*!
  @brief モデルを書き出す
  */
std::string FeatureClassifierDecisionStumpForJointBoostParams::print()const{
	std::stringstream ss;
	ss << std::fixed << threshold << "," << class_num << "," << focusing_feature << ",";
	ss << std::fixed << a << "," << std::fixed << b << "," << k.size() << ",";
	for(size_t c=0;c<k.size();c++){
		ss << std::fixed <<  k[c] << ",";
	}
	ss << subset_bitflag;;
	return ss.str();
}

/*!
  @brief モデルを読み込む
  @param filename 読み込み先の文字列
  */
void FeatureClassifierDecisionStumpForJointBoostParams::scan(const std::string& content){
	StringSplitter ssplitter(',');
	std::vector<std::string> buf;
	ssplitter.apply(content,&buf);

	assert(buf.size()>5);
	threshold = atof(buf[0].c_str());
//	std::cerr << "threshold: " << threshold << std::endl;
	class_num = atoi(buf[1].c_str());
//	std::cerr << "class_num: " << class_num << std::endl;

	focusing_feature = static_cast<size_t>(atoi(buf[2].c_str()));
//	std::cerr << "forcusing_feature: " << focusing_feature << std::endl;
	a = atof(buf[3].c_str());
//	std::cerr << "a+b: " << a << std::endl;
	b = atof(buf[4].c_str());
//	std::cerr << "b: " << b << std::endl;
	size_t k_size = atoi(buf[5].c_str());
	assert(buf.size()== 7 + k_size);
	size_t offset = 6;
	if(k_size>0){
		k.resize(k_size,0);
		for(size_t c=0;c<class_num;c++){
			k[c] = atof(buf[offset+c].c_str());
		}
	}
	offset += k_size;

	stringstream ss;
	ss.str(buf[offset]);
	ss >> subset_bitflag;
//	std::cerr << "subset_bit_flag: " << subset_bitflag << std::endl;
}

/*!
 * @brief threshold に値をセットするAccessor
 * @param threshold セットしたい値
 */
void FeatureClassifierDecisionStumpForJointBoostParams::setThreshold(double threshold ){
        this->threshold = threshold;
}

/*!
 * @brief threshold の値を取得する
 */
double FeatureClassifierDecisionStumpForJointBoostParams::getThreshold()const{
        return threshold;
}

/*!
 * @brief a の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setA(double a){
	this->a = a;
}

/*!
 * @brief a の値を取得する
 * */
double FeatureClassifierDecisionStumpForJointBoostParams::getA()const{
	return a;
}
/*!
 * @brief b の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setB(double b){
	this->b = b;
}

/*!
 * @brief b の値を取得する
 * */
double FeatureClassifierDecisionStumpForJointBoostParams::getB()const{
	return b;
}
/*!
 * @brief k の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setK(const std::vector<double>& k){
	this->k = k;
}

/*!
 * @brief k の値を取得する
 * */
double FeatureClassifierDecisionStumpForJointBoostParams::getK(size_t idx)const{
	return k[idx];
}

/*!
 * @brief subset_bitflag の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setSubsetBitflag(const Bitflag& subset_bitflag){
	this->subset_bitflag = subset_bitflag;
}

/*!
 * @brief subset_bitflag の値を取得する
 * */
const Bitflag& FeatureClassifierDecisionStumpForJointBoostParams::getSubsetBitflag()const{
	return subset_bitflag;
}

/*!
 * @brief class_num の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setClassNum(unsigned int class_num){
	this->class_num = class_num;
}

/*!
 * @brief class_num の値を取得する
 * */
unsigned int FeatureClassifierDecisionStumpForJointBoostParams::getClassNum()const{
	return class_num;
}


/*!
 * @brief focusing_feature の値をセットする
 * */
void FeatureClassifierDecisionStumpForJointBoostParams::setFocusingFeature(size_t focusing_feature){
	this->focusing_feature = focusing_feature;
}

/*!
 * @brief focusing_feature の値を取得する
 * */
size_t FeatureClassifierDecisionStumpForJointBoostParams::getFocusingFeature()const{
	return focusing_feature;
}




/**** Classifier ****/

/*!
 * @brief デフォルトコンストラクタ
 */
FeatureClassifierDecisionStumpForJointBoost::FeatureClassifierDecisionStumpForJointBoost():
	_error(DBL_MAX){
}

/*!
 * @brief コピーコンストラクタ
 * */
FeatureClassifierDecisionStumpForJointBoost::FeatureClassifierDecisionStumpForJointBoost(const FeatureClassifierDecisionStumpForJointBoost& other):
	FeatureClassifierWithParam<FeatureClassifierDecisionStumpForJointBoostParams>(other)
{
};

/*!
 * @brief デストラクタ
 */
FeatureClassifierDecisionStumpForJointBoost::~FeatureClassifierDecisionStumpForJointBoost(){
}

/*!
  @brief 学習アルゴリズム
  */
bool FeatureClassifierDecisionStumpForJointBoost::train(const std::vector<Feature>& trainSet,const MmplVector* weight){
	double target_class_weight_sum;
	return train(trainSet,weight,NULL,&target_class_weight_sum,NULL);
}

bool FeatureClassifierDecisionStumpForJointBoost::train(
		const std::vector<Feature>& trainSet,
		const MmplVector* weight,
		FeatureClassifierDecisionStumpForJointBoostTrainSampleData* train_set_data,
		double* target_class_weight_sum,
		FeatureClassifierDecisionStumpForJointBoostIntermidiateData* current_node,
		std::vector<Bitflag>* hasSelected){
	unsigned int class_num = param.getClassNum();
	assert(class_num <= JOINT_BOOST_MAX_CLASS_NUM);
	size_t sample_num = trainSet.size();
	if(sample_num==0){
		std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
		std::cerr << "Error: empty training_set." << std::endl;
		return false;
	}

	assert(weight!=NULL);
	assert(static_cast<unsigned int>(weight->size())==sample_num * class_num);

	size_t feature_dim = trainSet[0].size();
	assert(feature_dim>0);

	std::vector<std::vector<size_t> > __SortingMap,__SortingMap_inv;
	std::vector<std::vector<size_t> >* SortingMap, *SortingMap_inv;
	std::vector<std::vector<double> >* thresh_candidates(NULL);


	if(train_set_data==NULL){
		SortingMap = &__SortingMap;
		SortingMap_inv = &__SortingMap_inv;
	}
	else{
		SortingMap = &(train_set_data->sorting_map);
		SortingMap_inv = &(train_set_data->sorting_map_inv);
		thresh_candidates = &(train_set_data->thresh_candidates);
		train_set_data->feature_dim = feature_dim;
	}



	// 並び替え順序が未計算ならば、ここで計算する
	if(SortingMap->empty()){
		if(!calcSortingMap(trainSet,SortingMap,thresh_candidates)){
			std::cerr << "at " << __FILE__ <<": " << __LINE__ << std::endl;
			std::cerr << "Warning: Failed to calculate Sorting Map." << std::endl;
			return false;
		}
		invertSortingMap(*SortingMap,SortingMap_inv);
	}

	if(thresh_candidates!=NULL && current_node!=NULL){
		current_node->resize(thresh_candidates->size());
		for(size_t i=0;i<thresh_candidates->size();i++){
			current_node->resize(i,thresh_candidates->at(i).size());
		}
	}

	assert(SortingMap->size()==feature_dim);

	Bitflag bitflag_inv(param.getSubsetBitflag());
	bitflag_inv.flip();


	double threshold = -DBL_MAX;
	double min_error = DBL_MAX;
	size_t arg_min_feature = 0;
	double a=0;
	double b=0;
	for(size_t d = 0; d < feature_dim; d++){
		if(hasSelected!=NULL && 
				(hasSelected->at(d) & param.getSubsetBitflag()).any()){
			continue;
		}
		if(getThresholdAndMinError(
					trainSet,
					SortingMap->at(d),
					d,
					*weight,
					param.getSubsetBitflag(),
					&min_error,
					&threshold,
					&a,
					&b,
					target_class_weight_sum,
					current_node
					)){
			arg_min_feature = d;
		}
	}

	double weight_normalizing_factor = 0.0;
	for(size_t c = 0;c<class_num;c++){
		if(param.getSubsetBitflag()[c]){
			weight_normalizing_factor += train_set_data->default_total_weight[c];
		}
	}
	*target_class_weight_sum /= weight_normalizing_factor;

	this->_error = min_error;
	if(min_error == DBL_MAX){
		return true;
	}
	param.setThreshold(threshold);
	param.setFocusingFeature(arg_min_feature);
	param.setA(a);
	param.setB(b);
	calcRegressionParams(trainSet, weight);

	return true;
}

bool FeatureClassifierDecisionStumpForJointBoost::train(
		const std::vector<Feature>& trainSet,
		const MmplVector* weight,
		const FeatureClassifierDecisionStumpForJointBoostTrainSampleData& train_set_data,
		double* target_class_weight_sum,
		std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>* parent_nodes,
		FeatureClassifierDecisionStumpForJointBoostIntermidiateData* intermidiate_data,
		std::vector<Bitflag>* hasSelected
		){
	assert(intermidiate_data!=NULL);
	assert(parent_nodes!=NULL);

	size_t feature_dim = train_set_data.feature_dim;
	// 定数となる回帰の補正値を計算し、_errorにd,thetaによらない
	// エラーのoffsetを格納する
	_error = 0.0;
	std::vector<std::pair<Bitflag,Bitflag> > flag_combi;
	for(std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>::const_iterator ifb1 = parent_nodes->begin();
			ifb1 != parent_nodes->end(); ifb1++){
		std::map<Bitflag,FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>::const_iterator ifb2 = ifb1;
		ifb2++;
		for(;ifb2 != parent_nodes->end(); ifb2++){
			flag_combi.push_back(std::pair<Bitflag,Bitflag>(ifb1->first,ifb2->first));
		}
	}
	size_t class_num = param.getClassNum();

	std::vector<double> base_errors(flag_combi.size(),0.0);
	calcRegressionParamsParallel(trainSet,weight,flag_combi,&base_errors);


	std::vector<double> _weight_a_positive(class_num,0.0);
	std::vector<double> _weight_a_negative(class_num,0.0);

	for(size_t i=0;i<trainSet.size();i++){
		size_t offset = i * class_num;
		size_t class_id = trainSet[i].getTeacherClassID();
		for(size_t c=0;c<class_num;c++){
			double dw = weight->at(offset + c);
			if(c==class_id){
				_weight_a_positive[c] += dw;
			}
			else{
				_weight_a_negative[c] += dw;
			}
		}
	}


	std::vector<size_t> target_class_list1, target_class_list2;
	double min_error = DBL_MAX;
	size_t argmin_d = UINT_MAX;
	size_t argmin_combi = UINT_MAX;
	size_t argmin_thresh_index = UINT_MAX;

	tbb::task_scheduler_init TbbInit;
	std::vector<double> min_errors(flag_combi.size(),DBL_MAX);
	std::vector<size_t> argmin_thresh_indice(flag_combi.size(),UINT_MAX);
	std::vector<size_t> argmin_ds(flag_combi.size(),UINT_MAX);

	for(size_t d = 0; d < feature_dim; d++){

		std::cerr << "\b\b\b\b\b\b\b\b\b\b\b\bdim: " << (d+1) << "/" << feature_dim << std::flush;
		std::vector<double> weight_a_positive(_weight_a_positive);
		std::vector<double> weight_a_negative(_weight_a_negative);
		std::vector<double> weight_b_positive(class_num,0.0);
		std::vector<double> weight_b_negative(class_num,0.0);

		double prev_thresh = trainSet[train_set_data.sorting_map[d][0]][d];
		size_t s = 0;

		for(size_t x = 0; x < trainSet.size(); x++){
			size_t i = train_set_data.sorting_map[d][x];
			size_t class_id = trainSet[i].getTeacherClassID();

			TbbInit.initialize();
			tbb::parallel_for(
					tbb::blocked_range<size_t>(0,class_num),
					RegressionStumpUpdateWeightSequence(
						class_id,
						&(weight->at(i * class_num)),
						&weight_a_positive,
						&weight_a_negative,
						&weight_b_positive,
						&weight_b_negative),
					tbb::auto_partitioner());
			TbbInit.terminate();

			// 同じ値が続く間はエラーの評価を行わない
			if(trainSet[i][d] == prev_thresh) continue;
			prev_thresh = trainSet[i][d];

//			std::cerr << "step: " << s << std::endl;

			TbbInit.initialize();
			tbb::parallel_for(
					tbb::blocked_range<size_t>(0,flag_combi.size()),
					RegressionStumpNodeTrain(
						class_num,
						s,
						i,
						d,
						hasSelected,
						&flag_combi,
						&weight_a_positive,
						&weight_a_negative,
						&weight_b_positive,
						&weight_b_negative,
						parent_nodes,
						&min_errors,
						&argmin_thresh_indice,
						&argmin_ds
						),
					tbb::auto_partitioner());
			TbbInit.terminate();

			s++;
		}
	}
	std::cerr << std::endl;
	for(size_t combi = 0;combi < min_errors.size(); combi++){
		if(min_errors[combi] < 0 || min_errors[combi]>FLT_MAX) continue;
		if(min_errors[combi] + base_errors[combi] < min_error){
			min_error = min_errors[combi] + base_errors[combi];
			argmin_d = argmin_ds[combi];
			argmin_combi = combi;
			argmin_thresh_index = argmin_thresh_indice[combi];
		}
	}
	this->_error = min_error;
	if(min_error == DBL_MAX || min_error < 0){
		std::cerr << "No more selectable features." << std::endl;
		return true;
	}

	Bitflag newbit = flag_combi[argmin_combi].first | flag_combi[argmin_combi].second;
	std::cerr << "Add   : " << newbit << std::endl;

	*target_class_weight_sum = 0;
	size_t subset_class_num = 0;
	double weight_normalizing_factor = 0;
	for(size_t c = 0; c < class_num; c++){
		if(!newbit[c]) continue;
		subset_class_num++;
		*target_class_weight_sum += (_weight_a_positive[c] + _weight_a_negative[c]);
		weight_normalizing_factor += train_set_data.default_total_weight[c];
	}
	*target_class_weight_sum /= weight_normalizing_factor;

	// calc intermidiate data for newbit
	intermidiate_data->resize(feature_dim);

	Bitflag p1 = flag_combi[argmin_combi].first;
	Bitflag p2 = flag_combi[argmin_combi].second;

	intermidiate_data->b.resize(feature_dim);
	intermidiate_data->a_plus_b.resize(feature_dim);

	std::map<Bitflag, FeatureClassifierDecisionStumpForJointBoostIntermidiateData,BitflagComparer>::iterator pdata1,pdata2;
	pdata1 = parent_nodes->find(flag_combi[argmin_combi].first);
	assert(pdata1 != parent_nodes->end());
	pdata2 = parent_nodes->find(flag_combi[argmin_combi].second);
	assert(pdata2 != parent_nodes->end());

	size_t argmin_thresh_next = 0;
	for(size_t d = 0; d < feature_dim; d++){
		double w_a_pos1 = 0;
		double w_a_neg1 = 0;
		double w_a_pos2 = 0;
		double w_a_neg2 = 0;
		for(size_t c=0;c<class_num;c++){
			if(p1[c]){
				w_a_pos1 += _weight_a_positive[c];
				w_a_neg1 += _weight_a_negative[c];
			}
			else if(p2[c]){
				w_a_pos2 += _weight_a_positive[c];
				w_a_neg2 += _weight_a_negative[c];
			}
		}
		double w_b_pos1 = 0;
		double w_b_neg1 = 0;
		double w_b_pos2 = 0;
		double w_b_neg2 = 0;

		size_t s = 0;
		double prev_thresh = trainSet[train_set_data.sorting_map[d][0]][d];
		for(size_t x = 0;x<trainSet.size();x++){
			size_t i = train_set_data.sorting_map[d][x];
			size_t class_id = trainSet[i].getTeacherSignal();
			for(size_t c = 0;c<class_num;c++){
				if( !p1[c] && !p2[c] ) continue;
				double dw = weight->at(i * class_num + c);
				if(c==class_id){
					if(p1[c]){
						w_a_pos1 -= dw;
						w_b_pos1 += dw;
					}
					else if(p2[c]){
						w_a_pos2 -= dw;
						w_b_pos2 += dw;
					}
				}
				else{
					if(p1[c]){
						w_a_neg1 -= dw;
						w_b_neg1 += dw;
					}
					else if(p2[c]){
						w_a_neg2 -= dw;
						w_b_neg2 += dw;
					}
				}
			}

			// 同じ値が続く間はエラーの評価を行わない
			if(trainSet[i][d] == prev_thresh) continue;
			prev_thresh = trainSet[i][d];

			double b = pdata1->second.b[d][s] * (w_b_pos1 + w_b_neg1)
					 + pdata2->second.b[d][s] * (w_b_pos2 + w_b_neg2);
			intermidiate_data->b[d].push_back(b/(w_b_pos1 + w_b_neg1 + w_b_pos2 + w_b_neg2));

			double a_plus_b = pdata1->second.a_plus_b[d][s] * (w_a_pos1 + w_a_neg1)
							+ pdata2->second.a_plus_b[d][s] * (w_a_pos2 + w_a_neg2);
			intermidiate_data->a_plus_b[d].push_back(a_plus_b / (w_a_pos1 + w_a_neg1 + w_a_pos2 + w_a_neg2));

			if(i==argmin_thresh_index && d==argmin_d){
				param.setA(intermidiate_data->a_plus_b[d][s]);
				param.setB(intermidiate_data->b[d][s]);
				if(x!=trainSet.size()){
					argmin_thresh_next = train_set_data.sorting_map[d][x+1];
				}
			}
			s++;
		}
	}

	std::cerr << "Erase1: " << flag_combi[argmin_combi].first << std::endl;
	std::cerr << "Erase2: " << flag_combi[argmin_combi].second << std::endl;
	parent_nodes->erase(pdata1);
	parent_nodes->erase(pdata2);

	if(argmin_thresh_index == train_set_data.sorting_map.size()-1){
		param.setThreshold(trainSet[argmin_thresh_index][argmin_d]);
	}
	else{
		std::cerr << min_error << std::endl;
		param.setThreshold(
				( trainSet[argmin_thresh_index][argmin_d]
				+ trainSet[argmin_thresh_next ][argmin_d])/2
				);
	}
	param.setSubsetBitflag(newbit);
	param.setFocusingFeature(argmin_d);

	return true;
}

void FeatureClassifierDecisionStumpForJointBoost::calcRegressionParams(const std::vector<Feature>& trainSet,const MmplVector* weight){
	size_t class_num = param.getClassNum();
	// 回帰のための計算
	std::vector<double> k(class_num,0.0);
	std::vector<double> k_plus(class_num,0.0);
	std::vector<double> k_minus(class_num,0.0);


	double error_k = 0.0;

	std::vector<size_t> non_targets;
	for(size_t c=0;c<class_num;c++){
		if(!param.getSubsetBitflag()[c]){
			non_targets.push_back(c);
		}
	}

	for(size_t i=0;i<trainSet.size();i++){
		double v = trainSet[i][param.getFocusingFeature()];
		double thresh = param.getThreshold();
		unsigned int teacher_class_id = trainSet[i].getTeacherClassID();

		size_t offset = i * class_num;
		for(size_t j = 0; j < non_targets.size(); j++){
			size_t c = non_targets[j];
			double dw = weight->at(offset + c);
			k[c] += dw;
			if(c == teacher_class_id){
				k_plus[c] += dw;
			}
			else{
				k_minus[c] += dw;
			}
		}
	}

	double d_error = 0.0;
	for(size_t j=0;j<non_targets.size();j++){
		size_t c = non_targets[j];
		if(k[c]==0) continue;
		k[c] = (k_plus[c] - k_minus[c]) / k[c];
		d_error += std::pow( 1.0 - k[c],2.0 ) * k_plus[c];
		d_error += std::pow(-1.0 - k[c],2.0 ) * k_minus[c];
	}
	//	std::cerr << "d_error : " << d_error << std::endl;
	_error += d_error;
	param.setK(k);
}

void FeatureClassifierDecisionStumpForJointBoost::calcRegressionParamsParallel(
		const std::vector<Feature>& trainSet,
		const MmplVector* weight,
		const std::vector<std::pair<Bitflag,Bitflag> >& flag_combi,
		std::vector<double>* errors
		){
	size_t class_num = param.getClassNum();
	// 回帰のための計算
	std::vector<double> k(class_num,0.0);
	std::vector<double> k_plus(class_num,0.0);
	std::vector<double> k_minus(class_num,0.0);


	double error_k = 0.0;

	for(size_t i=0;i<trainSet.size();i++){
		double v = trainSet[i][param.getFocusingFeature()];
		double thresh = param.getThreshold();
		unsigned int teacher_class_id = trainSet[i].getTeacherClassID();

		size_t offset = i * class_num;
		for(size_t c = 0; c < class_num; c++){
			double dw = weight->at(offset + c);
			k[c] += dw;
			if(c == teacher_class_id){
				k_plus[c] += dw;
			}
			else{
				k_minus[c] += dw;
			}
		}
	}

	for(size_t c=0; c < class_num; c++){
		if(k[c]==0) continue;
		k[c] = (k_plus[c] - k_minus[c]) / k[c];
	}

	assert(errors->size() == flag_combi.size());

	for(size_t i=0;i<flag_combi.size();i++){
		Bitflag bitflag = flag_combi[i].first | flag_combi[i].second;
		for(size_t c=0;c<class_num;c++){
			if(!bitflag[c]) continue;
			errors->at(i) += std::pow( 1.0 - k[c],2.0 ) * k_plus[c];
			errors->at(i) += std::pow(-1.0 - k[c],2.0 ) * k_minus[c];
		}
	}
	//	std::cerr << "d_error : " << d_error << std::endl;
	param.setK(k);
}


/*
 * @brief 学習したモデルに従ってh_m(v,c)を返す。他のjointboostingの結果との比較はできないが、学習時の重みの再計算には必要
 * @param testSample テスト用サンプル
 * @param useBandK bとkを加味しないならfalseを入れる。この場合は単なるpredict関数と同じになる
 * */
double FeatureClassifierDecisionStumpForJointBoost::predict(Feature* testSample,bool useBandK)const{
	if(!useBandK){
		return predict(testSample);
	}
	assert(testSample!=NULL);
	assert(testSample->size()>0);

	assert( param.getFocusingFeature() < testSample->size());

	MmplVector likelihood(param.getClassNum(),0.0);
	for(unsigned int c = 0; c < param.getClassNum(); c++){
		// この弱識別器が対象としていないクラスにはKを返す
		if(!param.getSubsetBitflag()[c]){
			likelihood[c] = param.getK(c);
		}
		else if(testSample->at(param.getFocusingFeature()) > param.getThreshold() ){
			likelihood[c] = param.getA();
		}
		else{
			likelihood[c] = param.getB();
		}
	}

	testSample->setLikelihood(likelihood);
	return likelihood.max();
}



/*
   @brief 学習したモデルに従って識別を行う(bとkは他のjointboostingの出力と比較するために利用しない)
   */
double FeatureClassifierDecisionStumpForJointBoost::predict(Feature* testSample)const{
	assert(testSample!=NULL);
	assert(testSample->size()>0);

	assert( param.getFocusingFeature() < testSample->size());

	MmplVector likelihood(param.getClassNum(),0.0);
	for(unsigned int c = 0; c < param.getClassNum(); c++){
		// この弱識別器が対象としていないクラスには0を返す
		if(!param.getSubsetBitflag()[c]){
			likelihood[c] = 0.0;
		}
		else if(testSample->at(param.getFocusingFeature()) > param.getThreshold()){
			likelihood[c] = param.getA();
		}
		else{
			likelihood[c] = param.getB();
		}
	}

	testSample->setLikelihood(likelihood);
	return likelihood.max();
}

FeatureClassifierDecisionStumpForJointBoostParams& FeatureClassifierDecisionStumpForJointBoostParams::operator=(const FeatureClassifierDecisionStumpForJointBoostParams& other){
	threshold = other.threshold;
	a = other.a;
	b = other.b;
	k = other.k;
	subset_bitflag = other.subset_bitflag;
	class_num = other.class_num;
	focusing_feature = other.focusing_feature;
	return *this;
}

/*
 * @brief ソート結果をインデックスとして返す
 * @param trainSet ソート対象のサンプルセット
 * @param SortingMap ソート結果。例えばd次元目の特徴量でソート結果が一番目のものへはtrainSet[SortingMap[d][0]][d]でアクセスできる。
 * */
bool FeatureClassifierDecisionStumpForJointBoost::calcSortingMap(
				const std::vector<Feature>& trainSet,
				std::vector<std::vector<size_t> >* SortingMap,
				std::vector<std::vector<double> >* thresh_candidates)const{
	size_t sample_num = trainSet.size();
	size_t feature_dim = trainSet[0].size();

	for(size_t i = 0;i < sample_num; i++){
		if(trainSet[i].size() != feature_dim){
			std::cerr << "at " << __FILE__ << ": " << __LINE__ << std::endl;
			std::cerr << "Warning: invalid feature size at " << i << "th sample. Feature dimension of the first sample was " << feature_dim << "." << std::endl;
			std::cerr << trainSet[i] << std::endl;
			return false;
		}
	}

	SortingMap->assign(feature_dim,std::vector<size_t>(sample_num,0));
	if(thresh_candidates!=NULL){
		thresh_candidates->assign(feature_dim,std::vector<double>());
	}

	for(size_t d = 0;d < feature_dim; d++){
		// use std::pair and set to sort object with its index.
		std::set<std::pair<double,size_t> > sorting_set;
		for(size_t i=0; i<sample_num; i++){
			sorting_set.insert(std::pair<double,size_t>(trainSet[i][d],i));
		}
		std::set<std::pair<double,size_t> >::iterator it;
		size_t idx = 0;
		for(it=sorting_set.begin();it!=sorting_set.end();it++,idx++){
//			std::cerr << it ->first << " ( " << it->second << ")" << std::endl;
			SortingMap->at(d)[idx] = it->second;
		}

		if(thresh_candidates==NULL) continue;

		std::set<double> thresh_set;
		for(size_t i=0; i<sample_num;i++){
			thresh_set.insert(trainSet[i][d]);
		}
		for(std::set<double>::iterator it = thresh_set.begin();it != thresh_set.end(); it++){
			thresh_candidates->at(d).push_back(*it);
		}
//		std::cerr << d << ": " << thresh_set.size() << std::endl;
	}
//	std::cerr << trainSet.size() << std::endl;
//	std::cerr << sample_num << std::endl;
	return true;
}

void FeatureClassifierDecisionStumpForJointBoost::invertSortingMap(
		const std::vector<std::vector<size_t> >& SortingMap,
		std::vector<std::vector<size_t> >* SortingMap_inv)const{
	assert(SortingMap_inv != NULL);
	*SortingMap_inv = SortingMap;
	for(size_t d = 0; d < SortingMap.size();d++){
		for(size_t i = 0; i < SortingMap[d].size(); i++){
			SortingMap_inv->at(d)[i] = SortingMap[d][SortingMap[d].size()-1-i];
		}
	}
}


bool FeatureClassifierDecisionStumpForJointBoost::getThresholdAndMinError(
	const std::vector<Feature>& trainSet,
	const std::vector<size_t>& SortingMap,
	size_t d,
	const std::vector<double>& weight,
	const Bitflag& bitflag,
	double* _min_error,
	double* threshold,
	double* _arg_min_a,
	double* _arg_min_b,
	double* target_class_weight_sum,
	FeatureClassifierDecisionStumpForJointBoostIntermidiateData* node
	)const{

	double error = 0;
	size_t class_num = weight.size() / trainSet.size();

	std::vector<size_t> target_class_list;
	for(size_t c = 0;c<class_num;c++){
		if(bitflag[c]){
			target_class_list.push_back(c);
		}
	}

	double min_error = DBL_MAX;
	size_t arg_min_error = 0;
	double thresh_buf(-DBL_MAX);

	double w_a_all = 0.0;
	double w_a_plus = 0.0;
	double w_a_minus = 0.0;
	double w_b_all = 0.0;
	double w_b_plus = 0.0;
	double w_b_minus = 0.0;


	for(size_t x=0;x<trainSet.size();x++){
		size_t i = SortingMap[x];
		size_t offset = i * class_num;
		size_t class_id = trainSet[i].getTeacherClassID();
		for(size_t c=0;c<target_class_list.size();c++){
			double dw = weight[offset + target_class_list[c]];
			w_a_all += dw;
			if(target_class_list[c]==class_id){
				w_a_plus += dw;
			}
			else{
				w_a_minus += dw;
			}
		}
	}

	*target_class_weight_sum = w_a_all;
	double a = (w_a_plus - w_a_minus)/w_a_all;
	double b = 0;
	double arg_min_a = a;
	double arg_min_b = 0;

	size_t j = 0;
	for(size_t x=0;x<trainSet.size();x++){
		if(w_a_minus == 0.0 || w_a_plus == 0.0) break;

		size_t i = SortingMap[x];
		size_t class_id = trainSet[i].getTeacherClassID();


		size_t weight_offset = i * class_num;
		for(size_t c = 0; c < target_class_list.size();c++){
			double dw = weight[weight_offset + target_class_list[c]];
			if(dw == 0.0) continue;
			w_a_all -= dw;
			w_b_all += dw;
			if(target_class_list[c] == class_id){
				w_a_plus -= dw;
				w_b_plus += dw;
			}
			else{
				w_a_minus -= dw;
				w_b_minus += dw;
			}
		}

		// 同じ値が続く間はエラーの評価を行わない
		if(x==0){
			thresh_buf = trainSet[i][d];
			continue;
		}
		if(trainSet[i][d] == thresh_buf) continue;

		assert(thresh_buf < trainSet[i][d]);

		thresh_buf = trainSet[i][d];

		a = (w_a_all==0) ? 0 : (w_a_plus - w_a_minus)/w_a_all;
		b = (w_b_all==0) ? 0 : (w_b_plus - w_b_minus)/w_b_all;
		error =   std::pow( 1.0 - a, 2.0) * (w_a_plus)
				+ std::pow( -1.0 - a,2.0) * (w_a_minus)
				+ std::pow( 1.0 - b, 2.0) * (w_b_plus)
				+ std::pow( -1.0 - b,2.0) * (w_b_minus);

/*		error = std::pow(1.0-a,2.0) * w_a_all;
			+	std::pow(1.0-b,2.0) * w_b_all;
*/
		if(error < min_error){
			min_error = error;
			arg_min_error = i;
			arg_min_a = a;
			arg_min_b = b;
		}

		if(node==NULL) continue;

/*		if(j>=node->b[d].size()){
			for(size_t y = 0; y < trainSet.size();y++){
				size_t i = SortingMap[x];
			}

			std::cerr << d << ": " << x << "/" << trainSet.size() << std::endl;
			std::cerr << d << ": " << j << std::endl;
			std::cerr << d << ": " << node->b.size() << std::endl;
			std::cerr << d << ": " << node->b[d].size() << std::endl;
			std::cerr << d << ": " << (-0.0 == 0.0) << std::endl;
			std::cerr << d << ": " << node->b[d][0] << "..." << node->b[d][node->b[d].size()-1] << std::endl;
			std::cerr << d << ": " << trainSet[i][d] << std::endl;
		}
*/
		assert(j < node->b[d].size());
		node->b[d][j] = b;
		node->a_plus_b[d][j] = a;
		j++;
	}

	if(*_min_error <= min_error){
		return false;
	}

	*_min_error = min_error;
	if(arg_min_error == SortingMap.size()-1){
		*threshold = trainSet[arg_min_error][d];
	}
	else{
		*threshold = (
				trainSet[arg_min_error    ][d] +
				trainSet[arg_min_error + 1][d] )/2;
	}
	*_arg_min_a = arg_min_a;
	*_arg_min_b = arg_min_b;
	return true;
}

double FeatureClassifierDecisionStumpForJointBoost::getError()const{
	return _error;
}

void FeatureClassifierDecisionStumpForJointBoost::getTrainSampleData(
		const std::vector<Feature>& trainSet,
		const MmplVector& weight,
		FeatureClassifierDecisionStumpForJointBoostTrainSampleData* train_set_data)const{
	assert(train_set_data!=NULL);
	assert(!trainSet.empty());
	calcSortingMap(trainSet,&(train_set_data->sorting_map),&(train_set_data->thresh_candidates));
//	invertSortingMap(train_set_data->sorting_map,&(train_set_data->sorting_map_inv));
	train_set_data->feature_dim = trainSet[0].size();

	size_t class_num = weight.size() / trainSet.size();
	train_set_data->default_total_weight.resize(class_num,0.0);
	for(size_t i=0;i<weight.size();i++){
		size_t c = i % class_num;
		train_set_data->default_total_weight[c] += weight[i];
	}

}

} // namespace mmpl
