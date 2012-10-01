/*!
 * @file VideoCaptureFlyCapture.h
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/18
 * @date Last Change:2012/Oct/01.
 */
#ifndef __SKL_VIDEO_CAPTURE_FLY_CAPTURE_H__
#define __SKL_VIDEO_CAPTURE_FLY_CAPTURE_H__


// FlyCapture2
#include "FlyCapture2.h"

// SKL OpenCV Module
#include "sklcv.h"

// other files in FlyCapture Module
#include "sklFlyCapture2_utils.h"


namespace skl{
	class FlyCapture;
	/*!
	 * @brief FlyCapture2を使って撮影を行うVideoCaptureClass
	 */
	class VideoCaptureFlyCapture: public VideoCaptureInterface<VideoCaptureFlyCapture>{
		friend class FlyCapture;
		public:
			using _VideoCaptureInterface::set;
			using _VideoCaptureInterface::get;
			VideoCaptureFlyCapture(cv::Ptr<FlyCapture2::BusManager> busMgr=NULL);
			virtual ~VideoCaptureFlyCapture();
			virtual bool open(int device);
			virtual bool isOpened()const{return is_opened;}
			virtual void release();
			virtual bool grab();
			virtual bool retrieve(cv::Mat& image, int channel=0);

			virtual bool set(capture_property_t prop_id,double val);
			virtual double get(capture_property_t prop_id);

			FlyCapture2::Camera* getCamera(){return &camera;}
		protected:
			bool is_opened;
			bool is_started;
			FlyCapture2::Camera camera;
			FlyCapture2::Image flycap_image;

			cv::Ptr<FlyCapture2::BusManager> busMgr;

			bool set_flycap(capture_property_t prop_id,double val);
			bool set_for_develop(capture_property_t prop_id, double val);
			double get_flycap(capture_property_t prop_id);
			double get_for_develop(capture_property_t prop_id);
			static bool isFlyCapProperty(capture_property_t prop_id);

			static std::map<capture_property_t, FlyCapture2::PropertyType> prop_type_map;
		private:
			static void initialize_prop_type_map();
			virtual bool open(const std::string& filename);
			static FlyCapture2::FrameRate getFrameRate(double fps);
			inline FlyCapture2::FrameRate getFrameRate(){
				return getFrameRate(get(skl::FPS));
			}
			static FlyCapture2::VideoMode getVideoMode(int width,int height);
			inline FlyCapture2::VideoMode getVideoMode(){
				return getVideoMode(
							get(skl::FRAME_WIDTH),
							get(skl::FRAME_HEIGHT));
			}
	};

} // skl

#endif // __SKL_VIDEO_CAPTURE_FLY_CAPTURE_H__

