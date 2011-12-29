How to use

Encapsulate.rb hoge.h
hoge.h中のクラス内に以下のフォーマットで宣言されている変数_varを自動的にカプセル化し、
標準出力に書き出す
--- 前略 ---
public:
	int _var; //> get,set
--- 後略 ---
必要な条件
1. 変数名が'_'で始まる
2. 同じ行に "//>" で始まるコメントがある
オプション
get: 値を読む関数 int var()const を作成する
set: 値をセットする関数 void var(int __var) を作成する

例えば次のクラスは以下のように出力される。
---- 入力ファイル (test.h) ----
#include "hoge.h"

class Test{
	public:
		const std::vector<int> _var; //>get
		std::string _huga; //>set
		double _hoge; //>get,set
};
---- 出力 ----
% ./Encapsulate.rb test.h
#include "hoge.h"

class Test{
	public:
		const std::vector<int>& var()const{return _var;}
		void huga(const std::string& __huga){_huga = __huga;}
		double hoge()const{return _hoge;}
		void hoge(double __hoge){_hoge = __hoge;}
	protected:
		const std::vector<int> _var; 
		std::string _huga; 
		double _hoge; 
};

