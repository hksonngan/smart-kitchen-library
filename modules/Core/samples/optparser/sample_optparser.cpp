#include <iostream>
#include <vector>
#include "OptParser.h"

opt_on(int, foo, 10, "-f","<INT>", "set parameter foo.");
opt_on_bool(verbose, "", "an example of bool switch.");

// declare std::vector option manually for flexible initialization
std::vector<double> vec(3,0.0);
opt_on_container(std::vector,double, vec,"-v","<D:D:D>", "set double vector", ":", 3);

int main(int argc,char* argv[]){
	skl::OptParser options;
	std::vector<std::string> args;
	opt_parse(options,argc,argv,args);
	if(options.help()){
		options.usage();
		return -1;
	}

	for(size_t i = 0; i < args.size(); i++){
		std::cout << "argv[" << i << "] = " << args[i] << std::endl;
	}


	std::cout << "int foo = " << foo << std::endl;
	std::cout << "bool verbose = " << verbose << std::endl;
	std::cout << "vec =";
	for(size_t i=0;i<vec.size();i++){
		std::cout << " " << vec[i];
	}
	std::cerr << std::endl;

	return 0;
}

