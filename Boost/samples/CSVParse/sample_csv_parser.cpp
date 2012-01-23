#include <iostream>
#include "sklboost.h"

int main(int argc, char* argv[]){
	if(argc<2){
		std::cerr << "Usage: " << argv[0] << " file.csv" << std::endl;
		return 0;
	}

	std::vector<std::vector<std::string> > csv_data;
	csv_parse(argv[1],&csv_data);

	for(size_t j = 0; j < csv_data.size(); j++){
		for(size_t i = 0; i < csv_data[j].size(); i++){
			std::cout << "\"" << csv_data[j][i] << "\" ";
		}
		std::cout << std::endl;
	}

	return 1;
}
