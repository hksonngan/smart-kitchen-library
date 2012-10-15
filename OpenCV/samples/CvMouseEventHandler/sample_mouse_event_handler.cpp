/*!
 * @file sample_mouse_data.cpp
 *
 * @author 橋本敦史
 * @date Last Change: 2012/Oct/15.
 * */

#include "sklcv.h"

#include <iostream>
#include <csignal>

void printEvent(const skl::MouseEvent& event);
void printFlag(int flag);

int main(int argc,char **argv){

	cv::Mat canvas(cv::Size(1024,768), CV_8UC3, cv::Scalar(255,255,255));

	// GUI
	cv::namedWindow("canvas",0);

	// マウスイベントを取得する関数on_mouseの設定
	skl::CvMouseEventHandler mouse_on_canvas("canvas");

	// KeyInput
	skl::KeyInput key;

	cv::Point mouse_location;

	// ここから画像読み込み・表示部分
	while('q'!=key.set( cvWaitKey(10) ) ){
		if(key.getChar()=='c'){
			canvas =  cv::Mat(canvas.size(),canvas.type(),cv::Scalar(255,255,255));
		}

		// print mouse location
		if(mouse_location != mouse_on_canvas.location()){
			// current mouse location
			std::cerr << "Mouse is on (" << mouse_on_canvas.location().x << ", " << mouse_on_canvas.location().y << ")." << std::endl;
			// current mosue flag
			printFlag(mouse_on_canvas.flag());
			mouse_location = mouse_on_canvas.location();
		}


		// print mouse event
		while(!mouse_on_canvas.events().empty()){
			skl::MouseEvent event = mouse_on_canvas.events().front();
			printEvent(event);
			printFlag(event.flag);
			mouse_on_canvas.events().pop();
		}

		//! マウスで指定した範囲の色を出力
		cv::imshow("canvas",canvas);
	}

	cv::destroyWindow("canvas");
	return EXIT_SUCCESS;
}

#define caseEventType(type) \
	case CV_EVENT_##type: \
		std::cout << #type;\
		break

void printEvent(const skl::MouseEvent& event){
	std::cout << "EventType: ";
	switch(event.type){
		case CV_EVENT_LBUTTONDOWN:
			std::cout << "LBUTTON_DOWN";
			break;
		// do similar process (by macro function)
		caseEventType(RBUTTONDOWN);
		caseEventType(MBUTTONDOWN);
		caseEventType(LBUTTONUP);
		caseEventType(RBUTTONUP);
		caseEventType(MBUTTONUP);
		caseEventType(LBUTTONDBLCLK);
		caseEventType(RBUTTONDBLCLK);
		caseEventType(MBUTTONDBLCLK);
		default:
			std::cout << "UNKNOWN";
	}
	std::cout << std::endl;

}


#define _printFlag(_flag)\
	std::cout << #_flag << ": " << ((flag&CV_EVENT_##_flag) > 0) << std::endl

void printFlag(int flag){
	std::cout << "FLAG_LBUTTON" << ": " << ((flag&CV_EVENT_FLAG_LBUTTON)>0) << std::endl;
	// do similar process (by macro function)
	_printFlag(FLAG_RBUTTON);
	_printFlag(FLAG_MBUTTON);
	_printFlag(FLAG_CTRLKEY);
	_printFlag(FLAG_SHIFTKEY);
//	_printFlag(FLAG_ALTKEY); // FLAG_ALTKEY conflict with NumLock. (this looks a bug in OpenCV)
}
