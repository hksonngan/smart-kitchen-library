/*!
 * @file FlyCapture.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change:2012/Jan/19.
 */
#ifndef __SKL_FLY_CAPTURE_H__
#define __SKL_FLY_CAPTURE_H__

// OpenCV
#include <cv.h>
#include <highgui.h>

// FlyCapture2
#include "FlyCapture2.h"

// SKL OpenCV Module
#include "sklcv.h"

// SKL FlyCap Files
#include "sklFlyCapture2_utils.h"
#include "VideoCaptureFlyCapture.h"

namespace skl{

	/*!
	 * @class FlyCaptureライブラリを用いて、複数カメラを同期撮影するVideoCapture
	 */
	class FlyCapture: public skl::VideoCapture{
		public:
			using skl::VideoCapture::open;
			using skl::VideoCapture::push_back;
			FlyCapture();
			virtual ~FlyCapture();
			virtual bool open();
			virtual bool grab();
			virtual void release();

		protected:
			bool is_started;

			bool sync_capture_start(FlyCapture2::Camera** ppCameras);

			// PC毎に共通のBusManager
			static FlyCapture2::BusManager* busMgr;

		private:
			// 同期撮影を専門とするため、
			// device番号の指定なしで、全てのカメラが得られる
			// 従ってdeviceを指定するカメラのopenは不可とする
			virtual bool open(int device){return false;}
			virtual bool push_back(int device){return false;}
			std::vector<VideoCaptureFlyCapture*> fcam_interface;
	};

} // skl

#endif // __SKL_FLY_CAPTURE_H__

