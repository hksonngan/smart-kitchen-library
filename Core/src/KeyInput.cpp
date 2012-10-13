/*!
 * @file KeyInput.cpp
 *
 * @author 橋本敦史
 * @date Last Change:2012/Oct/13.
 * */

#include "KeyInput.h"

using namespace std;

using namespace skl;
	int KeyInput::SHIFT = 0x10000;
	int KeyInput::CTRL = 0x40000;
	int KeyInput::NUMLOCK = 0x100000;
	int KeyInput::CAPSLOCK = 0x20000;
	int KeyInput::ALT = 0x80000;

	int KeyInput::LEFTARROW  = 65361;
	int KeyInput::UPARROW    = 65362;
	int KeyInput::RIGHTARROW = 65363;
	int KeyInput::DOWNARROW  = 65364;


	/*
	 * @brief コンストラクタ
	 * */
	KeyInput::KeyInput(){
		valid = false;
	}

	/*
	 * @brief デストラクタ
	 * */
	KeyInput::~KeyInput(){}

	/*
	 * @brief cvWaitKeyを通して入力されたキーを解析する
	 * @param input cvWaitKey()の返り値
	 * @return shift等を無視した文字だけの返り値(shift等の情報も欲しければgetKey()を使う)
	 * */
	int KeyInput::regulate(int input){
		if(input==-1){
			key.key = -1;
			// 何も入力されていないので、validにfalseを入れておく。
			valid = false;
		}
		else{
			key.numlock = input / NUMLOCK == 0 ? false : true;
			if(key.numlock)	input -= NUMLOCK;
			
			key.alt = input / ALT == 0 ? false : true;
			if(key.alt)	input -= ALT;

			key.ctrl = input / CTRL == 0 ? false : true;
			if(key.ctrl)	input -= CTRL;

			key.capslock = input / CAPSLOCK == 0 ? false : true;
			if(key.capslock)	input -= CAPSLOCK;

			key.shift = input / SHIFT == 0 ? false : true;
			if(key.shift)	input -= SHIFT;
			
			valid = true;
			key.key = input;
		}
		return key.key;
	}

	/*
	 * @brief 正統な入力が行われたどうかをチェック
	 * @return 入力の正統性
	 * */
	bool KeyInput::isValid(){
		return valid;
	}

	/*
	 * @brief メンバkeyを参照する
	 * @return メンバkeyの参照
	 * */
	const KeyData* KeyInput::getKey()const{
		return &key;
	}

	/*
	 * @brief 解析結果をstring型として出力
	 * @return 解析結果
	 * */
	string KeyInput::print(){
		stringstream buf;
		if(31 < key.key && key.key < 128){
			buf << "KEY(ch) : " << (char)key.key << endl;
		}
		else{
			buf << "KEY(int): " << (int)key.key << endl;
		}
		buf << "SHIFT   : " << key.shift << endl;
		buf << "CTRL    : " << key.ctrl << endl;
		buf << "NUMLOCK : " << key.numlock << endl;
		buf << "CAPSLOCK: " << key.capslock << endl;
		buf << "ALT     : " << key.alt << endl;
		return buf.str();
	}

