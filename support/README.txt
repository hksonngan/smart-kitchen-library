How to use

Encapsulate.rb hoge.h
hoge.h��Υ��饹��˰ʲ��Υե����ޥåȤ��������Ƥ����ѿ�_var��ưŪ�˥��ץ��벽����
ɸ����Ϥ˽񤭽Ф�
--- ��ά ---
public:
	int _var; //> get,set
--- ��ά ---
ɬ�פʾ��
1. �ѿ�̾��'_'�ǻϤޤ�
2. Ʊ���Ԥ� "//>" �ǻϤޤ륳���Ȥ�����
���ץ����
get: �ͤ��ɤ�ؿ� int var()const ���������
set: �ͤ򥻥åȤ���ؿ� void var(int __var) ���������

�㤨�м��Υ��饹�ϰʲ��Τ褦�˽��Ϥ���롣
---- ���ϥե����� (test.h) ----
#include "hoge.h"

class Test{
	public:
		const std::vector<int> _var; //>get
		std::string _huga; //>set
		double _hoge; //>get,set
};
---- ���� ----
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

