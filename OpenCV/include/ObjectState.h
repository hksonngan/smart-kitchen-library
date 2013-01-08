/*!
 * @file ObjectState.h
 * @author a_hasimoto
 * @date Date Created: 2013/Jan/08
 * @date Last Change:2013/Jan/08.
 */
#ifndef __SKL_OBJECT_STATE_H__
#define __SKL_OBJECT_STATE_H__


#include "Flow.h"
#define SKL_LOCATION_RESOLUTION_DEFAULT 512

namespace skl{

/*!
 * @class 机上物体の状態を保持する値型オブジェクト
 * @note 得に意味的なつながりは無いが、データ構造が似ているのでオプティカルフロークラスを継承して関数を流用
 */
 class ObjectState: public Flow{
	public:
		enum HandlingState{
			NOT_HANDLED=0,
			HANDLED = 1
		};

		ObjectState(int location_resolution=SKL_LOCATION_RESOLUTION_DEFAULT);
		virtual ~ObjectState();

		inline cv::Point argmax_location(HandlingState h)const{
			float prob = -FLT_MAX;
			return argmax_location(h,prob);
		}

		inline cv::Point argmax_location()const{
			float prob = -FLT_MAX;
			return argmax_location(prob);
		}

		cv::Point argmax_location(HandlingState h, float& prob)const;
		cv::Point argmax_location(float& prob)const;
		float probHandlingState(HandlingState h)const;

		int resolution()const;

		inline const cv::Mat& location(HandlingState h_state)const{
			assert(0<=h_state && h_state<2);
			if(h_state==HANDLED) return u;
			return v;
		}

		inline cv::Mat& location(HandlingState h_state){
			assert(0<=h_state && h_state<2);
			if(h_state==HANDLED) return u;
			return v;
		}

		// 念のため、Flowのオペレータをオーバライドしておく.location()を使っておけば
		// 得に問題はない
		inline const cv::Mat& operator[](HandlingState h)const{return location(h);}
		inline cv::Mat& operator[](HandlingState h){return location(h);}

	protected:
	private:
		
};

} // skl

#endif // __SKL_OBJECT_STATE_H__

