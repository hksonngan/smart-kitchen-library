/*!
 * @file sample_keyinput.cpp
 *
 * @author 橋本敦史
 * @date Last Change: 2012/Oct/14.
 * */

#include "sklcv.h"

int main(int argc,char **argv){

	// prepare a window for get key input via cv::waitKey() function.
	cv::namedWindow("key input window",0);

	// prepare an instance for parsing key input.
	skl::KeyInput key;

	while('q'!= key.set( cvWaitKey(10) ) ){
		if(!key.isValid()) continue;

		// print key input
		std::cout << "Char:    " << key.getChar() << "(" << (int)key.getChar() << ")" << std::endl;
		std::cout << "Code:    " << key.code() << std::endl;
		std::cout << "Shift:   " << key.shift() << std::endl;
		std::cout << "Ctrl:    " << key.ctrl() << std::endl;
		std::cout << "Alt:     " << key.alt() << std::endl;
		std::cout << "NumLock: " << key.numlock() << std::endl;
		std::cout << "CapsLock:" << key.capslock() << std::endl;

		std::cout << "ARROW   :";
		switch(key.code()){
			case skl::KeyInput::LEFT_ARROW:
				std::cout << "LEFT";
				break;
			case skl::KeyInput::UP_ARROW:
				std::cout << "UP";
				break;
			case skl::KeyInput::RIGHT_ARROW:
				std::cout << "RIGHT";
				break;
			case skl::KeyInput::DOWN_ARROW:
				std::cout << "DOWN";
				break;
			default:
				std::cout << "-";
		}
		std::cout << std::endl;
	}

	cvDestroyWindow("key input window");
	return EXIT_SUCCESS;
}

