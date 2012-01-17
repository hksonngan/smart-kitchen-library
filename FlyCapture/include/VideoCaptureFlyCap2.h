/*!
 * @file FlyCapture.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change:2012/Jan/17.
 */
#ifndef __SKL_VIDEO_CAPTURE_FLYCAP2_H__
#define __SKL_VIDEO_CAPTURE_FLYCAP2_H__

#include <cv.h>
#include <highgui.h>
#include "FlyCapture2.h"

#include "sklcv.h"

namespace skl{

	class FlyCapture;
	/*!
	 * @class FlyCapture用のカメラパラメタ用インタフェイス. ただしOpenCVとの互換性からAbsControlが可能なものは強制的に絶対値でのパラメタ入出力となる
	 * */
	class FlyCaptureCameraInterface: public VideoCaptureCameraInterface{
		public:
			FlyCaptureCameraInterface(FlyCapture* fly_capture,int cam_id);
			~FlyCaptureCameraInterface();
			virtual double get(capture_property_t prop_id);
			virtual bool set(capture_property_t prop_id,double val);
		protected:

			double get_for_develop(capture_property_t prop_id);
			double get_flycap(capture_property_t prop_id);

			bool set_for_develop(capture_property_t prop_id,double val);
			bool set_flycap(capture_property_t prop_id,double val);

			bool isFlyCapProperty(capture_property_t prop_id);

			FlyCapture* fly_capture;
			static std::map<capture_property_t, FlyCapture2::PropertyType> prop_type_map;
			static void initialize_prop_type_map();
	};

	/*!
	 * @class FlyCaptureライブラリを用いて、複数カメラを同期撮影するVideoCapture
	 */
	class FlyCapture: public skl::VideoCapture{
		friend class FlyCaptureCameraInterface;
		public:
			FlyCapture();
			virtual ~FlyCapture();

			virtual bool open(int device);
			virtual bool open(const std::string& filename);
			virtual bool isOpened()const;
			virtual void release();
			virtual bool grab();
			virtual bool retrieve(cv::Mat& image,int channel=0);
			virtual VideoCapture& operator>> (cv::Mat& image);

		protected:
			bool is_file_loading;
			bool is_opened;
			bool is_started;

			bool initialize(int device);
			bool sync_capture_start();
			bool grab_flycap();
			bool develop(cv::Mat& image, int device);


			// PC毎に共通のBusManager
			static FlyCapture2::BusManager* busMgr;

			// PC毎に共通のカメラの数
			unsigned int numCameras;
			std::vector<FlyCapture2::CameraInfo> camInfo;
			std::vector<FlyCapture2::Image> camImages;
			FlyCapture2::Camera** ppCameras;

			bool checkError(FlyCapture2::Error& error,const char* file, int line);
			static void PrintBuildInfo();
			void PrintCameraInfo(int device)const;

		private:
	};

} // skl

#endif // __SKL_VIDEO_CAPTURE_FLYCAP2_H__

