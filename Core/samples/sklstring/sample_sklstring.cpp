#include <iostream>
#include <sstream>
#include <vector>
#include "sklstring.h"

std::string print_vec(const std::vector<std::string>& vec);
int main(int argc,char* argv[]){

	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " any_string [...]" << std::endl;
		return -1;
	}

	for(int i=1;i<argc;i++){
		std::cout << "=== TEST FOR ARGV[" << i << "] ===" << std::endl;
		std::cout << "input = \"" << argv[i] << "\"" << std::endl;
		std::cout << "skl::split(<input>,\"::\") = " << print_vec(skl::split(argv[i],"::")) << std::endl;
		std::cout << "skl::split(<input>,\"::\",2) = " << print_vec(skl::split(argv[i],"::",2)) << std::endl;
		std::cout << "skl::strip(<input>) = \"" << skl::strip(argv[i]) << "\"" << std::endl;
		std::cout << std::endl;
	}

	std::vector<std::string> join_material(argc-1);
	for(int i=1;i<argc;i++){
		join_material[i-1] = argv[i];
	}

	std::cout << "=== TEST FOR JOIN ALL ARGV === " << std::endl;
	std::cout << "skl::join(argv,\"+\") = \"" << skl::join(join_material, "+") << "\"" << std::endl;

	return 0;
}

std::string print_vec(const std::vector<std::string>& vec){
	std::stringstream ss;
	for(size_t i=0;i<vec.size();i++){
		if(i!=0){
			ss << "\" and ";
		}
		ss << "\"" << vec[i];
	}
	ss << "\"";
	return ss.str();
}

