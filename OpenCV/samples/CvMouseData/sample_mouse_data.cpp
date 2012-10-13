/*!
 * @file MultiFunctionalFileCameraViewer.cpp
 *
 * @author 橋本敦史
 * @date Last Change: 2010/Jul/24.
 * */

#include "ImageImage.h"
#include "ImageCameraFileCamera.h"
#include "CvMouseData.h"
#include "KeyInput.h"
#include "ParamReader.h"

#include <iostream>
#include <csignal>

void usage(const std::string& command);
void printSelectedRegionColors(const mmpl::image::Image& frame,const CvRect& selected_area);
mmpl::image::camera::FileCamera* createCamera(const mmpl::ParamReader& args);
void draw_rect(const CvRect& rect, const CvScalar& color, mmpl::image::Image* img);

int main(int argc,char **argv){
	mmpl::ParamReader args(argc,argv);

	if(args.getArgSize()<2 || args.hasHelp()){
		usage(args[0]);
		return EXIT_FAILURE;
	}

	mmpl::image::camera::FileCamera *cam;
	cam = createCamera(args);

	mmpl::image::Image frame;
	int index=0;
	int step=0;
	int last_index=-1;

	//! GUIの設定
	cvInitSystem(argc,argv);
	cvNamedWindow("frame",0);
	cvCreateTrackbar("INDEX","frame",&index,cam->getLength()-1,NULL);
	cvCreateTrackbar("STEP","frame",&step,256,NULL);

	//! キーボードの入力を処理するクラス
	mmpl::KeyInput key;

	//! マウスイベントを取得する関数on_mouseの設定
	mmpl::CvMouseData md;
	cvSetMouseCallback ("frame", mmpl::onMouse, (void *)&md);

	CvRect selected_area;
	bool hasSelectedRegion = false;
	CvScalar red = CV_RGB(255,0,0);
	CvScalar blue = CV_RGB(0,0,255);

	//! ここから画像読み込み・表示部分
	while('q'!=(char)( key.regulate( cvWaitKey(10) ) ) ){
		//! キーボードに入力があった場合、内容を出力
		if(key.isValid()){
			if(key.CTRL && 's'==key.getKey()->key){
				std::cerr << "保存先のファイル名を入力してください:" << std::endl;
				std::string filename;
				std::cin >> filename;
				frame.saveImage(filename);
			}
		}

		//! キャプチャする
		cam->capture(&frame,index);
		if(index!=last_index){
			std::cout << cam->getCurrentFileName() << std::endl;
			last_index=index;
		}

		//! マウスで指定した範囲の色を出力
		switch(md.popEvent()){
			case CV_EVENT_LBUTTONDOWN:	
				selected_area.x = md.eventPoint.x;
				selected_area.y = md.eventPoint.y;
				hasSelectedRegion=true;
				std::cerr << __LINE__ << ": " << selected_area.x << ", " << selected_area.y << std::endl;
				break;
			case CV_EVENT_MBUTTONDOWN:
				break;
			case CV_EVENT_RBUTTONDOWN:
				break;
			case CV_EVENT_LBUTTONUP:
				selected_area.width = md.eventPoint.x - selected_area.x;
				if(selected_area.width<0){
					selected_area.width *= -1;
					selected_area.x -=selected_area.width;
				}
				selected_area.height = md.eventPoint.y - selected_area.y;
				if(selected_area.height<0){
					selected_area.height *= -1;
					selected_area.y -= selected_area.height;
				}
				//! 描画色
				draw_rect(selected_area,red,&frame);
				printSelectedRegionColors(frame,selected_area);
				hasSelectedRegion=false;
				break;
			case CV_EVENT_RBUTTONUP:
				break;
			case CV_EVENT_MBUTTONUP:
				break;
			case CV_EVENT_LBUTTONDBLCLK: //! OpenCV側で未実装(reserved)
				break;
			case CV_EVENT_RBUTTONDBLCLK: //! OpenCV側で未実装(reserved)
				break;
			case CV_EVENT_MBUTTONDBLCLK:
				break;
		}

		if(hasSelectedRegion==true){
			selected_area.width = md.mousePoint.x - selected_area.x;
			if(selected_area.width<0){
				selected_area.width *= -1;
				selected_area.x -=selected_area.width;
			}
			selected_area.height = md.mousePoint.y - selected_area.y;
			if(selected_area.height<0){
				selected_area.height *= -1;
				selected_area.y -= selected_area.height;
			}
			//! 描画色
			draw_rect(selected_area,blue,&frame);
		}
		index+=step;
		if(static_cast<unsigned int>(index)>=cam->getLength()){
			continue;
		}

		//! トラックバーの移動
		cvSetTrackbarPos("INDEX","frame",index);

		//! 画像の表示
		cvShowImage("frame",frame.getIplImage());
	}

	delete cam;
	cvDestroyWindow("frame");
	return EXIT_SUCCESS;
}

mmpl::image::camera::FileCamera* createCamera(const mmpl::ParamReader& args){
	if(!args["bayer_pattern"].empty() && args["bayer_pattern"]=="BGGR"){
		return new mmpl::image::camera::FileCamera(args[1],mmpl::image::Image::BAYEREDGESENSE,mmpl::image::Image::BAYER_PATTERN_BGGR);
	}
	else if(!args["bayer_pattern"].empty() && args["bayer_pattern"]=="RGGB"){
		return new mmpl::image::camera::FileCamera(args[1],mmpl::image::Image::BAYEREDGESENSE,mmpl::image::Image::BAYER_PATTERN_RGGB);
	}
	else if(!args["bayer_pattern"].empty() && args["bayer_pattern"]=="GRBG"){
		return new mmpl::image::camera::FileCamera(args[1],mmpl::image::Image::BAYEREDGESENSE,mmpl::image::Image::BAYER_PATTERN_GRBG);
	}
	else if(!args["bayer_pattern"].empty() && args["bayer_pattern"]=="GBRG"){
		return new mmpl::image::camera::FileCamera(args[1],mmpl::image::Image::BAYEREDGESENSE,mmpl::image::Image::BAYER_PATTERN_GBRG);
	}
	return new mmpl::image::camera::FileCamera(args[1],mmpl::image::Image::DEFAULT);
}

void draw_rect(const CvRect& rect, const CvScalar& color, mmpl::image::Image* img){
	CvPoint cvPt1,cvPt2;
	cvPt1.x = rect.x;
	cvPt1.y = rect.y;
	cvPt2.x = rect.x + rect.width;
	cvPt2.y = rect.y + rect.height;
	cvRectangle(img->getIplImage(),cvPt1,cvPt2,color,1,CV_AA,0);
}

void printSelectedRegionColors(const mmpl::image::Image& frame,const CvRect& selected_area){
	std::string name;
	std::cerr << "Region Name: " << std::flush;
	std::cin >> name;
	for(int y=0;y<selected_area.height;y++){
		for(int x=0;x<selected_area.width;x++){
			mmpl::Color col = frame.getColor(selected_area.x+x,selected_area.y+y);
//			col = col.convColor(mmpl::Color::BGR);
			std::cout << name << ", FORMAT: BGR";
			for(int c=0;c<3;c++){
				std::cout << ", " << static_cast<int>(col[c]);
			}
			std::cout << std::endl;
		}
	}
}


void usage(const std::string& command){
	std::cerr << "Usage: " << command << " image.lst0" << std::endl;
	std::cerr << "--bayer_pattern=<PATTERN>: set bayer_pattern" << std::endl;
	std::cerr << "\t<PATTERN>: BGGR, RGGB, GRBG, GBRG" << std::endl;
}
