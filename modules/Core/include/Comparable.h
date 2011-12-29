/*!
 * @file Comparable.h
 * @author 橋本敦史
 * @date Last Change:2009/Dec/15 13:40:26.
 */
#ifndef __COMPARABLE_H__
#define __COMPARABLE_H__


namespace mmpl{

/*!
 * @class 比較可能なオブジェクトであることを示す(operator<を定義するだけで他の比較演算子が出来る) 
 */
template<class T> class Comparable{

	public:
		Comparable();
		virtual ~Comparable();
		// これだけは定義しなければならない純粋仮想関数
		virtual bool operator<(const T& other)const=0;
		friend bool operator>=(const T& lhs,const T& rhs){
			return !(lhs < rhs);
		}

		friend  bool operator>(const T& lhs,const T& rhs){
			return rhs < lhs;
		}

		friend  bool operator<=(const T& lhs,const T& rhs){
			return !(lhs > rhs);
		}

		friend  bool operator!=(const T& lhs,const T& rhs){
			return (lhs<rhs || lhs>rhs);
		}

		friend  bool operator==(const T& lhs,const T& rhs){
			return (lhs<=rhs && lhs>=rhs);
		}

	protected:
		
	private:
		
};

/*!
 * @brief デフォルトコンストラクタ
 */
template<class T> Comparable<T>::Comparable(){

}

/*!
 * @brief デストラクタ
 */
template<class T> Comparable<T>::~Comparable(){

}


} // namespace mmpl
#endif // __COMPARABLE_H__

