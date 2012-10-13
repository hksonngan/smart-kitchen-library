/*!
 * @file CvMouseData.cpp
 *
 * CvWindow上でのマウスイベントなどをMouseCallback関数の外に持ち出す変数とonMouse関数
 * @author 橋本敦史
 * @date Last Change:2012/Oct/13.
 * */

#include "CvMouseData.h"

namespace skl{

	/*!
	 * @brief コンストラクタ
	 * */
	CvMouseData::CvMouseData(){
		_event = 0;
		CvPoint zeropt;
		zeropt.x = 0;
		zeropt.y = 0;
		eventPoint = zeropt;
		mousePoint = zeropt;
		flagLbutton = false;
		flagRbutton = false;
		flagMbutton = false;
		flagCtrlKey = false;
		flagShiftKey = false;
		flagAltKey = false;
	}

	/*!
	 * @brief デストラクタ
	 * */
	CvMouseData::~CvMouseData(){}
	/*!
	 * @brief cvSetMouseCallbackの第二引数としてセットするコールバック関数
	 * @param event マウスイベント(OpenCV側から与えられる)
	 * @param x マウスのx座標(OpenCV側から与えられる)
	 * @param y マウスのy座標(OpenCV側から与えられる)
	 * @param flag マウスの状態フラグ(OpenCV側から与えられる)
	 * @param params 外部で定義されたobjectへのポインタ(ここではMouseDataを想定)
	 * */

	/*!
	 * @brief マウスイベントが起こったときに、その値を保持するクラス(一度読み取ると、イベントは0に初期化される)
	 * */
	int CvMouseData::popEvent(){
		int temp = _event;
		_event = 0;
		return temp;
	}

	void onMouse(int event, int x, int y, int flag, void *params){
		CvMouseData *md = (CvMouseData*)params;
		//! マウスイベントの獲得
		if(event!=CV_EVENT_MOUSEMOVE){
			md->_event = event;
			md->eventPoint.x = x;
			md->eventPoint.y = y;
			//			std::cout << "flag = " << flag << std::endl;
		}

		//! マウスの現在地を獲得
		md->mousePoint.x = x;
		md->mousePoint.y = y;

		//! フラグの値が64以上なら、それ以上のbitを無視
		if(flag>63){
			//			std::cout << "63 over: flag = " << flag << std::endl;
			flag%=64;
		}

		//! マウスのフラグを解析

		if(flag&32){
			//			std::cout << "alt: flag = " << flag << std::endl;
			//! Numlock? WebではAltキーとなっているが、Linuxではウィンドウ操作に割り当てられているため、確認不可
			md->flagAltKey = true;
		}
		else{
			md->flagAltKey = false;
		}

		if(flag&16){
			//			std::cout << "shift: flag = " << flag << std::endl;

			//! shift
			md->flagShiftKey = true;
		}
		else{
			md->flagShiftKey = false;
		}

		if(flag&8){
			//			std::cout << "ctrl: flag = " << flag << std::endl;
			//! ctrl
			md->flagCtrlKey = true;
		}
		else{
			md->flagCtrlKey = false;
		}

		if(flag&4){
			//			std::cout << "middle: flag = " << flag << std::endl;
			//! middle botton
			md->flagMbutton = true;
		}
		else{
			md->flagMbutton = false;
		}

		if(flag&2){
			//			std::cout << "right: flag = " << flag << std::endl;
			//! right botton
			md->flagRbutton = true;
		}
		else{
			md->flagRbutton = false;
		}

		if(flag&1){
			//			std::cout << "left: flag = " << flag << std::endl;
			//! left botton
			md->flagLbutton = true;
		}
		else{
			md->flagLbutton = false;
		}

	}
}
