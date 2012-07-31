#ifndef __SKL_FLY_CAPTURE2_UTILS_H__
#define __SKL_FLY_CAPTURE2_UTILS_H__

// C++
#include <iostream>

// FlyCapture2
#include "FlyCapture2.h"

#define SKL_FLYCAP2_CHECK_ERROR(error) FlyCapture2CheckError(error,__FILE__,__LINE__)
namespace skl{

	bool FlyCapture2CheckError(FlyCapture2::Error& _error, const char* file, int line);

	void FlyCapture2PrintBuildInfo();

	/*!
	 * @brief FlyCapture§Œ•´•·•È§À¥ÿ§π§ÅE Û§ÚΩ–Œœ§π§ÅE	 * */
	void FlyCapture2PrintCameraInfo(const FlyCapture2::CameraInfo& camInfo);
}

#endif // __SKL_FLY_CAPTURE2_UTILS_H__
