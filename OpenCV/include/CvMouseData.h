/*!
 * @file CvMouseData.h
 *
 * CvWindow上でのマウスイベントなどをMouseCallback関数の外に持ち出す変数とonMouse関数
 * @author 橋本敦史
 * @date Last Change:2012/Oct/13.
 * */

#ifndef __SKL_CV_MOUSE_DATA__
#define __SKL_CV_MOUSE_DATA__

#include "cv.h"
#include "highgui.h"

namespace skl{

	void onMouse(int event, int x,int y,int flag,void* params);

	/*
	 * @class OpenCV上でのマウス操作を取得するためのクラス
	 * */
	class CvMouseData{
		friend void onMouse(int event, int x,int y,int flag,void* params);

		public:
			CvMouseData();
			~CvMouseData();

			int popEvent();

			//! 直前に起きたMouseMove以外のイベントの発生場所を一時的に保存
			CvPoint eventPoint;

			//! 現在のマウスポインタの位置
			CvPoint mousePoint;

			//! ctrl,shif.altいずれも押さずにマウスの右ボタンが押下されたときtrue
			bool flagLbutton;

			//! ctrl,shif.altいずれも押さずにマウスの右ボタンが押下されたときtrue
			bool flagRbutton;

			//! ctrl,shif.altいずれも押さずにマウスの中央ボタンが押下されたときtrue
			bool flagMbutton;

			//! キーボードのctrlキーを押した状態でマウスのボタンを離したときと、その直後にtrue
			bool flagCtrlKey;

			//! キーボードのshiftキーを押した状態でマウスのボタンを離したときと、その直後にtrue
			bool flagShiftKey;

			//! キーボードのaltキーを押した状態でマウスのボタンを離したときと、その直後にtrue. しかし、Numlockとコンフリクトするバグがあるようなので使わない方が無難
			bool flagAltKey;
		private:
			//! 直前に起きたMouseMove以外のイベントの番号を格納 CV_EVENT_xxx に対応(http://opencv.jp/opencv/document/opencvref_highgui_simple.html#decl_cvSetMouseCallback)
			int _event;


	};

} // namespace skl

#endif // __CV_MOUSE_DATA__

