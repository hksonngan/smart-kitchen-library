#!/usr/bin/ruby

require 'optparse'
require 'date'
require 'erb'

TEMPLATE_DIR = "#{File.dirname(__FILE__)}/templates/"

opt = OptionParser.new

output_dir_include = "./"
output_dir_src = "./"
opt.on("-m DIR","--module DIR","Set OutputModule"){|v|
	if !File.exist?(v) or !File.directory?(v) then
		STDERR.puts "ERROR: Directory '#{v}' is not exist."
		exit -1
	end

	output_dir_include = "#{v}/include"
	if !File.exist?(output_dir_include) or !File.directory?(output_dir_include) then
		output_dir_include = v
	end
	output_dir_src = "#{v}/src"
	if !File.exist?(output_dir_src) or !File.directory?(output_dir_src) then
		output_dir_src = v
	end
}
author = `whoami`.strip
opt.on("-a VAL","--author VAL","Set Author Name"){|v|
	author = v
}


namespace = []
using_namespace = []
opt.on("-n VAL","--namespace VAL","Set NameSpace by namespace1::namespace2::..."){|v|
	namespace = v.split("::")
	for i in 0..namespace.size-1 do
		using_namespace << namespace[0..i].join("::")
	end
}

baseclass_name = []
baseclass_filename = []
opt.on("-p VAL","--parent VAL","Set Parent Class (by base1,base2 when two or more)."){|v|
	baseclass_name = v.split(",")

	for file in baseclass_name do
		_file = file.gsub(/<.*>/,"")
		buf = _file.split("::")
		classname = buf[buf.size-1]
		namespacebuf = ""
		if buf.size > 1 then
			namespacebuf = buf[0..buf.size-2].each{|v| v.capitalize!}.join("")
		end
		# remove template
		_file = namespacebuf + classname + ".h"
		baseclass_filename << _file
	end
}
argv = opt.parse!(ARGV)

if argv.size < 1 then
	STDERR.puts "Usage: #{__FILE__} class_name [file_basename] [options]"
	exit -1
end

class_name = argv[0]
file_basename = class_name
if argv.size > 1 then
	file_basename = argv[1]
end

header_file = "#{output_dir_include}/#{file_basename}.h"
cpp_file = "#{output_dir_src}/#{file_basename}.cpp"

date = Time::now.strftime("%Y/%b/%d")

# remove redundant namespace for baseclass
for i in 0..baseclass_name.size-1 do
	bn = baseclass_name[i]
	for ns in using_namespace.reverse do
		ns += "::"
		if bn.include?(ns) then
			baseclass_name[i] = bn.sub(ns,"")
			break
		end
	end
end


if File.exist?(header_file) then
	STDERR.puts "#{header_file} already exists.\n"
	exit -1
end
if File.exist?(cpp_file) then
	STDERR.puts "#{cpp_file} already exists.\n"
	exit -1
end

# make #ifndef MACRO
macro_for_prevent_dup=""
if !namespace.empty? then
	for ns in namespace do
		macro_for_prevent_dup += ns.split(/(?![a-z])(?=[A-Z])/).each{|elem| elem.upcase!}.join("_") + "_"
	end
end
macro_for_prevent_dup += file_basename.split(/(?![a-z])(?=[A-Z])/).each{|elem| elem.upcase!}.join("_")

baseclass_filename_include = ""
for bf in baseclass_filename do
	baseclass_filename_include += "#include \"#{bf}\"\n"
end

namespace_begin = ""
namespace.each_index do |i|
	ns = namespace[i]
	namespace_begin += "#{"\t"*i}namespace #{ns}{\n"
end

namespace_end = ""
namespace.reverse.each_index do |i|
	ns = namespace[i]
	namespace_end += "#{"\t"*(namespace.size-1-i)}} // #{ns}\n"
end

using_namespace.each_index do |i|
	using_namespace[i] = "using namespace #{using_namespace[i]};\n"
end

dot = "."
h_template=ERB.new(open("#{TEMPLATE_DIR}/H_template.erb").read)
cpp_template=ERB.new(open("#{TEMPLATE_DIR}/CPP_template.erb").read)

puts "generate: #{header_file}"
puts "generate: #{cpp_file}"

File.open(header_file,"w").write h_template.result(binding)
File.open(cpp_file,"w").write cpp_template.result(binding)
