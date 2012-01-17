#include "VideoCaptureFlyCap2.h"
#include "FlyCapture2.h"
using namespace skl;

bool FlyCapture::checkError(FlyCapture2::Error& _error, const char* file, int line){
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

/*!
 * @brief FlyCaptureのライブラリに燗する情報を出力する
 * */
void FlyCapture::PrintBuildInfo()
{
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
 * @brief FlyCaptureのカメラに関する情報を出力する
 * */
void FlyCapture::PrintCameraInfo(int id)const
{
	if(numCameras <= static_cast<unsigned int>(id)){
		printf("No camera[%d] in FlyCapture.",id);
	}
	printf(
			"\n*** CAMERA INFORMATION ***\n"
			"Serial number - %u\n"
			"Camera model - %s\n"
			"Camera vendor - %s\n"
			"Sensor - %s\n"
			"Resolution - %s\n"
			"Firmware version - %s\n"
			"Firmware build time - %s\n\n",
			camInfo[id].serialNumber,
			camInfo[id].modelName,
			camInfo[id].vendorName,
			camInfo[id].sensorInfo,
			camInfo[id].sensorResolution,
			camInfo[id].firmwareVersion,
			camInfo[id].firmwareBuildTime );
}
