#!/usr/bin/ruby
require 'date'
require 'erb'
require 'set'
require 'optparse'
require 'pathname'

ModuleDependency = {}
require "#{File.dirname(__FILE__)}/DependencyList.rb"

# MakefileGenerator�Τ���ǥ��쥯�ȥ�Υѥ�
PathToGenerator = File::dirname(File.expand_path(__FILE__))
PathToTemplate = PathToGenerator + "/templates/"
temp = File::expand_path(PathToGenerator)

# ������ץȼ¹Ծ�꤫��skl�Υ롼�ȥǥ��쥯�ȥ�ؤ����Хѥ�
# MakefileGenerator(���Υե�����)��"�롼�ȥǥ��쥯�ȥ�/support"�β��ˤ���Ȳ��ꤷ�ư�ľ夬��
AbstPathToRoot = File.dirname(temp)
AbstPathToProjects = AbstPathToRoot

temp_array = []
Dir::foreach(AbstPathToProjects){|f|
	# �ǥ��쥯�ȥ����
	if FileTest.directory?(AbstPathToProjects + f) then
		# .�ǻϤޤ�ǥ��쥯�ȥ�����
		if f[0] != "."[0] then
			temp_array << f
		end
	end
}
ModuleList = temp_array.clone
temp_array.clear


# ���ޥ�ɥ饤������Υ��ץ�����ѡ����������
opt = OptionParser.new

# Author�򥳥ޥ�ɥ饤�󤫤��������
# ���ޥ�ɥ饤��λ��꤬�ʤ����
author = `whoami`.strip
# ���ޥ�ɥ饤��λ��꤬������
opt.on("-a VAL","--author VAL","Set Author Name. Default: current user name"){|v|
	author = v
}

output_dir = "./"
opt.on("-t OutputDir","--target OutputDir","Set output directory for the generated Makefile."){|v|
	output_dir = v
	if !File.directory?(output_dir) then
		STDERR.puts "WARNING: you cannot direct filename. -o/--output option can direct only directory name."
		output_dir = File.dirname(output_dir)
	end
}

# programname��Ʊ��ǥ��쥯�ȥ�ˤ��ꡢprogramname�Ȥ�
# �̤˥���ѥ��뤹��ɬ�פΤ���main�ؿ�������ʤ��ե����뤬����л��ꤹ��
option = []
opt.on("-o OPT_LIST","--option OPT_LIST","Set Optional cpp file names. e.g. -o TestClass or -o \"TestClassA TestClassB\""){|v|
	option = v.strip.split(" ")
}

argv = opt.parse!(ARGV)

def to_rel(base, target)
	sep = /#{File::SEPARATOR}+/o
		base = base.split(sep)
	target = target.split(sep)
	while base.first == target.first
		base.shift
		target.shift
	end
	File.join([".."]*base.size+target)
end
RelativePathToRoot = to_rel(File::expand_path(output_dir),AbstPathToRoot) + "/"
if argv.size<1 then
	STDOUT.puts "#{$0} execute_file_name Modules [OPTIONS]"
	STDOUT.puts "-h or --help for options."
	exit -1
end

execute_file_name = argv[0]

active_modules = Set.new
active_modules << "Core"
argv.delete_at 0

for mod in argv do
	for dep in ModuleDependency[mod] do
		active_modules << dep
	end
	active_modules << mod
end


# ������ڤ�Υ⥸�塼��Υꥹ��(��ʸ���Ⱦ�ʸ��)
sklModule = ""
sklmodule = ""
sklpublicmodule = ""
unknown_modules = []
# ModuleOrder��DependencyList.rb��Ǻ��������
# �ƥ⥸�塼��ΰ�¸�ط����¤�����
# (Core����˰��ֺǸ����� for gcc�λ���)
ModuleOrder.each{|mod|
	p mod
	if active_modules.include?(mod) then
		if sklModule!="" then
			sklModule+=" "
			sklmodule+=" "
		end
		sklModule += mod
		sklmodule += mod.downcase
	end
	if active_modules.include?(mod) and File.exist?("#{RelativePathToRoot}/public_modules/#{mod}") then
		if sklpublicmodule != "" then
			sklpublicmodule += " "
		end
		sklpublicmodule += mod.downcase
	end
}

for mod in active_modules do
	if !ModuleOrder.include?(mod) then
		STDERR.puts "#{mod} is not in dependency list. Please add it in the array 'ModuleOrder'of this script."
		exit 0
	end
end

if File.exist?("Makefile") then
	STDOUT.print "Makefile already exists. Are you sure to overwrite it?[y/n]: "
	STDOUT.flush
	ans = STDIN.gets.strip
	if ans!="y" then
		exit -1
	end
end

makefile_template = ERB.new(open(PathToTemplate+"Makefile_template.erb").read)

templates = Array.new
otheroptions = ""
active_modules.each{|mod|
	template_filename = PathToTemplate+"Makefile_" + mod.downcase + "_template.txt"
	if File.exist?(template_filename) then
		otheroptions += open(template_filename).read + "\n"
	end
}

open("Makefile","w") do |f|
	f.print makefile_template.result(binding)
end

print "Makefile was successfully generated.\n"
