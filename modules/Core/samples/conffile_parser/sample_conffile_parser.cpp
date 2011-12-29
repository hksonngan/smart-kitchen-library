#include <iostream>
#include "sklstring.h"

int main(int argc, char* argv[]){

	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " test.conf" << std::endl;
		return -1;
	}
	std::map<std::string,std::string> conf_params;
	if(!skl::parse_conffile(argv[1],conf_params)){
		std::cerr << "Error: Failed to open '" << argv[1] << std::endl;
	}

	// print
	for(std::map<std::string,std::string>::iterator iter = conf_params.begin();
			iter != conf_params.end(); iter++){
		std::cout << "conf_params[\"" << iter->first << "\"] = \"" << iter->second << "\"" << std::endl;
	}
	return 0;
}
