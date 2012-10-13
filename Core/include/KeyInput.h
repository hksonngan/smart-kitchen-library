/*!
 * @file KeyInput.h
 *
 * @author 橋本敦史
 * @date Last Change:2012/Oct/13.
 * */

#ifndef __SKL_KEY_INPUT__
#define __SKL_KEY_INPUT__

#include <string>
#include <sstream>

namespace skl{
	/*
	 * @struct キーボード入力の解析結果を格納する
	 * */
	struct KeyData{
		int key; //! キーボードへの入力結果からshift等の影響を取り除いた整数
		bool shift; //! shiftキーが押されていたかどうか
		bool ctrl; //! ctrlキーが押されていたかどうか
		bool numlock; //! numlockキーの状態
		bool capslock; //! capslockキーの状態
		bool alt; //! altキーが押されていたかどうか
	};

	/*
	 * @class cvWaitKeyで取得したキーを整理する
	 * */
	class KeyInput{
		public:
			KeyInput();
			~KeyInput();
			int regulate(const int input);
			const KeyData* getKey()const;
			static int SHIFT;
			static int CTRL;
			static int NUMLOCK;
			static int CAPSLOCK;
			static int ALT;
			static int RIGHTARROW;
			static int LEFTARROW;
			static int UPARROW;
			static int DOWNARROW;
			std::string print();
			bool isValid();
		private:
			KeyData key;
			bool valid;
	};
}

#endif // __KEY_INPUT__


