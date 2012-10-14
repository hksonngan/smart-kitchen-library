/*!
 * @file CvMouseData.h
 *
 * CvWindow上でのマウスイベントなどをMouseCallback関数の外に持ち出す変数とonMouse関数
 * @author 橋本敦史
 * @date Last Change:2012/Oct/14.
 * */

#ifndef __SKL_CV_MOUSE_DATA__
#define __SKL_CV_MOUSE_DATA__

#include "cv.h"
#include "highgui.h"
#include <queue>

namespace skl{

	struct MouseEvent{
		int type;
		int flag;
		cv::Point location;
	};

	/*
	 * @class OpenCV上でのマウス操作を取得するためのクラス
	 * */
	class CvMouseData{

		public:
			CvMouseData(const std::string& window_name);
			~CvMouseData();

			inline cv::Point location()const{return _location;}
			inline int flag()const{return _flag;}
			inline std::queue<MouseEvent>& events(){return _events;}
		protected:
			cv::Point _location;
			int _flag;
			std::queue<MouseEvent> _events;
		private:
			static void onMouse(int event, int x,int y,int flag,void* params);


	};

} // namespace skl

#endif // __CV_MOUSE_DATA__

