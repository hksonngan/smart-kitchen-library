/*!
 * @file CvMouseData.cpp
 *
 * OpenCV Window上でのマウスイベントをMouseCallback関数の外に持ち出す変数とonMouse関数
 * @author 橋本敦史
 * @date Last Change:2012/Oct/14.
 * */

#include "CvMouseData.h"

namespace skl{

	/*!
	 * @brief コンストラクタ
	 * */
	CvMouseData::CvMouseData(const std::string& win_name){
		cvSetMouseCallback (win_name.c_str(), this->onMouse, (void *)this);
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
	void CvMouseData::onMouse(int event_type, int x, int y, int flag, void *params){
		CvMouseData *md = (CvMouseData*)params;
		// マウスの現在地を獲得
		md->_location = cv::Point(x,y);

		// set flags
		md->_flag = flag;

		if(event_type == CV_EVENT_MOUSEMOVE) return;
		MouseEvent event;
		event.type = event_type;
		event.flag = flag;
		event.location = md->location();
		md->_events.push(event);
	}
}
