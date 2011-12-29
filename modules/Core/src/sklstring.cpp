#include "sklstring.h"
#include <cassert>
#include <iostream>
#include <fstream>

namespace skl{
	std::string strip(const std::string& str){
		if(str.empty()) return str;
		std::string dest;
		bool flag = false;
		size_t begin(0),end(str.size()-1);
		for(;begin<str.size();begin++){
			if(isgraph(str[begin])>0){
				break;
			}
		}
		assert(begin < str.size());

		for(;end>=0;end--){
			if(isgraph(str[end])>0){
				break;
			}
		}
		dest = str.substr(begin,end-begin+1);
		return dest;
	}

	std::vector<std::string> split(const std::string& str, const std::string& deliminator,int length){
		std::vector<std::string> buf;
		std::string temp = str;
		int idx;

		// for push_back out of "while"
		length--;

		while((idx=temp.find(deliminator)) != temp.npos && length != 0){
			buf.push_back(temp.substr(0,idx));
			temp = temp.substr(idx+deliminator.size());
			length--;
		}

		buf.push_back(temp);

		return buf;
	}

	std::string join(const std::vector<std::string>& buf, const std::string& separator){
		std::string str;
		for(size_t i=0;i<buf.size();i++){
			if(i!=0){
				str.insert(str.end(),separator.begin(),separator.end());
			}
			str.insert(str.end(),buf[i].begin(),buf[i].end());
		}
		return str;
	}


	std::vector<std::string> split_strip(const std::string& str, const std::string& deliminator, int length){
		std::vector<std::string> buf = split(str,deliminator,length);
		for(size_t i=0;i<buf.size();i++){
			buf[i] = strip(buf[i]);
		}
		return buf;
	}

	bool parse_conffile(const std::string& filename, std::map<std::string,std::string>& param_map,const std::string& deliminator){
		std::ifstream fin;
		fin.open(filename.c_str());
		if(!fin){
			return false;
		}
		std::vector<std::string> buf;
		std::string str;
		while(fin && std::getline(fin,str)){
			if(str.empty()) continue;
			str = strip(str);
			buf = split(str,"#",2);
			if(buf[0].empty()) continue;
			buf = split_strip(buf[0],deliminator,2);
			if(buf.size()<2 || buf[0].empty()){
				std::cerr << "WARNING: invalid format '" << str << "' in file '" << filename << "'." << std::endl;
				continue;
			}
			param_map[buf[0]] = buf[1];
		}
		fin.close();
		return true;
	}

}
