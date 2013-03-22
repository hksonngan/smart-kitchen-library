/*!
 * @file InterPolater.cpp
 * @author a_hasimoto
 * @date Date Created: 2013/Mar/22
 * @date Last Change: 2013/Mar/22.
 */
#include "InterPolater.h"

using namespace skl;


/*!
 * @brief コンストラクタ
 */
InterPolater::InterPolater(
				const std::vector<double>& x_values,
				const std::vector<double>& y_values,
				Border border):_border(border){
	setValues(x_values,y_values);
}

/*!
 * @brief コンストラクタ
 */
InterPolater::InterPolater(
				const double* x_values,
				const double* y_values,
				size_t value_num,
				Border border):_border(border){
	std::vector<double> __x_values(value_num);
	std::vector<double> __y_values(value_num);
	memcpy(&__x_values[0],x_values,value_num*sizeof(double));
	memcpy(&__y_values[0],y_values,value_num*sizeof(double));
	setValues(__x_values,__y_values);
}

/*!
 * @brief デフォルトコンストラクタ
 */
InterPolater::InterPolater(Border border):_border(border){

}


/*!
 * @brief デストラクタ
 */
InterPolater::~InterPolater(){

}

void InterPolater::setValues(
		const std::vector<double>& x_values,
		const std::vector<double>& y_values){
	assert(!x_values.empty());
	assert(x_values.size()==y_values.size());
	for(size_t i=1;i<x_values.size();i++){
		assert(x_values[i] > x_values[i-1]);
	}
	_x_values = x_values;
	_y_values = y_values;
}


double InterPolater::compute(double x_val)const{
	size_t num = _x_values.size();

	float y_val = _y_values[num-1];

	if(x_val<_x_values[0]){
		if(_border == CONSTANT){
			return _y_values[0];
		}
		float offset = _y_values[0];
		float div = (_y_values[1]-_y_values[0])/(_x_values[1]-_x_values[0]);
		y_val = (x_val-_x_values[0])*div + offset;
		return y_val;
	}

	for(size_t i=1;i<num;i++){
		if(x_val>_x_values[i] && (i!=num-1 || _border != EXTRAPOLATION)) continue;
		float offset = _y_values[i-1];
		float div = (_y_values[i]-_y_values[i-1])/(_x_values[i]-_x_values[i-1]);
		y_val = (x_val-_x_values[i-1])*div + offset;
		break;
	}
	return y_val;
}
