#!/usr/bin/ruby
require 'date'
require 'erb'
require 'set'
require 'optparse'
require 'pathname'

ModuleDependency = {}
require "#{File.dirname(__FILE__)}/DependencyList.rb"

# MakefileGeneratorのあるディレクトリのパス
PathToGenerator = File::dirname(File.expand_path(__FILE__))
PathToTemplate = PathToGenerator + "/templates/"
temp = File::expand_path(PathToGenerator)

# スクリプト実行場所からsklのルートディレクトリへの相対パス
# MakefileGenerator(このファイル)が"ルートディレクトリ/support"の下にあると仮定して一つ上がる
AbstPathToRoot = File.dirname(temp)
AbstPathToProjects = AbstPathToRoot

temp_array = []
Dir::foreach(AbstPathToProjects){|f|
	# ディレクトリだけ
	if FileTest.directory?(AbstPathToProjects + f) then
		# .で始まるディレクトリを除外
		if f[0] != "."[0] then
			temp_array << f
		end
	end
}
ModuleList = temp_array.clone
temp_array.clear


# コマンドライン引数のオプションをパースする準備
opt = OptionParser.new

# Authorをコマンドラインから取得する
# コマンドラインの指定がない場合
author = `whoami`.strip
# コマンドラインの指定がある場合
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

# programnameと同一ディレクトリにあり、programnameとは
# 別にコンパイルする必要のあるmain関数を持たないファイルがあれば指定する
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


# 空白区切りのモジュールのリスト(大文字と小文字)
sklModule = ""
sklmodule = ""
sklpublicmodule = ""
unknown_modules = []
# ModuleOrderはDependencyList.rb中で作成される
# 各モジュールの依存関係に並んだもの
# (Coreが常に一番最後に来る for gccの仕様)
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
