/*!
 * @file InterPolater.h
 * @author a_hasimoto
 * @date Date Created: 2013/Mar/22
 * @date Last Change:2013/Mar/22.
 */
#ifndef __SKL_INTER_POLATER_H__
#define __SKL_INTER_POLATER_H__

#include "skl.h"

namespace skl{

/*!
 * @class InterPolater
 * @brief 入力されたデータ点に従って、線形補完された値を返すクラス
 */
class InterPolater{

	public:
		enum Border{
			CONSTANT = 0,
			EXTRAPOLATION = 1
		};

		InterPolater(Border border=CONSTANT);
		InterPolater(
				const std::vector<double>& x_values,
				const std::vector<double>& y_values,
				Border border = CONSTANT);
		InterPolater(
				const double* x_values,
				const double* y_values,
				size_t value_num,
				Border border = CONSTANT);

		virtual ~InterPolater();

		double compute(double x_val)const;

		void setValues(
				const std::vector<double>& x_values,
				const std::vector<double>& y_values);

		inline void setBorder(Border border){
			_border = border;
		}
		inline Border getBorder()const{
			return _border;
		}

		inline const std::vector<double>& getXValues()const{
			return _x_values;
		}
		inline const std::vector<double>& getYValues()const{
			return _y_values;
		}

	protected:
		std::vector<double> _x_values;
		std::vector<double> _y_values;
		Border _border;
	private:
		
};

} // skl

#endif // __SKL_INTER_POLATER_H__

