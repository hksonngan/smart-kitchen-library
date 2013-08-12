/*!
 * @file sklFlyCapture2_utils.cpp
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/12
 * @date Last Change: 2012/Jan/18.
 */
#include "sklFlyCapture2_utils.h"
namespace skl{

	bool FlyCapture2CheckError(FlyCapture2::Error& _error, const char* file, int line){
		try{
			if(_error != FlyCapture2::PGRERROR_OK) throw(_error);
			return true;
		}
		catch( FlyCapture2::Error error ){
			std::cerr << "checked at File '" << file << "', line " << line << std::endl;
			error.PrintErrorTrace();
			return false;
		}
	}

	void FlyCapture2PrintBuildInfo(){
		FlyCapture2::FC2Version fc2Version;
		FlyCapture2::Utilities::GetLibraryVersion( &fc2Version );
		char version[128];
		sprintf( 
				version, 
				"FlyCapture2 library version: %d.%d.%d.%d\n", 
				fc2Version.major, fc2Version.minor, fc2Version.type, fc2Version.build );

		printf( "%s", version );

		char timeStamp[512];
		sprintf( timeStamp, "Application build date: %s %s\n\n", __DATE__, __TIME__ );

		printf( "%s", timeStamp );

	}

	/*!
	 * @brief FlyCapture¤Î¥«¥á¥é¤Ë´Ø¤¹¤EðÊó¤ò½ÐÎÏ¤¹¤E
	 * */
	void FlyCapture2PrintCameraInfo(const FlyCapture2::CameraInfo& camInfo){
		printf(
				"Serial number - %u\n"
				"Camera model - %s\n"
				"Camera vendor - %s\n"
				"Sensor - %s\n"
				"Resolution - %s\n"
				"Firmware version - %s\n"
				"Firmware build time - %s\n\n",
				camInfo.serialNumber,
				camInfo.modelName,
				camInfo.vendorName,
				camInfo.sensorInfo,
				camInfo.sensorResolution,
				camInfo.firmwareVersion,
				camInfo.firmwareBuildTime );
	}

}
