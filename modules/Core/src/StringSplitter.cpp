/*!
 * @file StringSplitter.cpp
 * @author 橋本敦史
 * @date Last Change:2010/Jun/18.
 */
#include "StringSplitter.h"
#include <sstream>

using namespace std;

namespace mmpl{

/*!
 * @brief デフォルトコンストラクタ
 */
StringSplitter::StringSplitter(char deliminator,bool validateDoubleQuot):
	deliminator(deliminator),
	validateDoubleQuot(validateDoubleQuot)
	{

}

/*!
 * @brief デストラクタ
 */
StringSplitter::~StringSplitter(){

}

/*
   * @brief 指定された区切り文字を利用して、文字列を分割する
   */
void StringSplitter::apply(const std::string& str,std::vector<std::string>* tar)const{

	stringstream ss;
	ss << str;
	
	

	tar->clear();

	string temp;
	bool isInDoubleQuot = false;
	string dquot_str;
	while(std::getline(ss,temp,deliminator)){
		if(!isInDoubleQuot){
			skipFrontSpace(&temp);
		}

		if(validateDoubleQuot){
			if(isInDoubleQuot){
				if(jointDoubleQuotElement(temp,&dquot_str)){
					tar->push_back(dquot_str);
					dquot_str.clear();
					isInDoubleQuot = false;
				}
				continue;
			}
			else if(temp.find("\"")==0){
				// ダブルクオートのはじまり
				// 次にダブルクォートが書いてある文字列が見付かるまで
				// tempに読み取った文字列+区切り文字を足しつづける
				if(jointDoubleQuotElement(temp.substr(1,temp.npos),&dquot_str)){
					tar->push_back(dquot_str);
					dquot_str.clear();
					continue;
				}
				isInDoubleQuot = true;
				continue;
			}
		}

		skipRearSpace(&temp);
		tar->push_back(temp);

	}

	if(!dquot_str.empty()){
		// "が閉じられていない場合の処理
		dquot_str.resize(dquot_str.size()-1);
		tar->push_back(dquot_str);
	}
}


bool StringSplitter::jointDoubleQuotElement(const std::string& input_str,std::string* dquot_str)const{
	// input_strの語尾に\"があれば、その直前までの文字列を
	// dquot_strに格納してtrueを返す。
	std::string str = input_str;
	skipRearSpace(&str);
	unsigned int back_dq = str.rfind("\"");
	if(str.size()>0 && back_dq==str.size()-1){
		// ダブルクオートがお尻にあるので、dquotに\"の前まで足してtrue
		*dquot_str += str.substr(0,back_dq);
		return true;
	}
	*dquot_str += input_str;
	dquot_str->resize(dquot_str->size()+1,deliminator);
	return false;

}

void StringSplitter::skipFrontSpace(std::string* str){
	int idx = 0;
	while((*str)[idx]==' '){
		idx++;
	}
	*str = str->substr(idx,str->npos);
}

void StringSplitter::skipRearSpace(std::string* str){
	int idx = 1;
	while((*str)[str->size()-idx]==' '){
		idx++;
	}
	*str = str->substr(0,str->size()-idx+1);

}


} // namespace skl


