#!/usr/bin/ruby

require 'optparse'

BasicTypes = ["int","double","short","size_t","float"]
for type in BasicTypes do
	if !type.include?("unsigned") then
		BasicTypes << "unsigned #{type}"
	end
end

opt = OptionParser.new

file_suffix = "_io"
opt.on("--file_suffix"){|v|
	file_suffix = v
}

argv = opt.parse!(ARGV)

if argv.size < 1 then
	STDERR.puts "Usage: #{__FILE__} header_file [options]"
	STDERR.puts "-h or --help for options"
	exit -1
end
HEADER_FILE = argv[0]


file_block=["",""]
file_block_idx = 0

get_params = []
set_params = []

def parse(line)
	type = ""
	var_name = ""
	_get = false
	_set = false
	if line =~ /(\s*)(.*)\s([a-zA-Z]\w*)\;\s*\/\/>(.*)/ then
		STDERR.puts "WARNING: \"//>\" is set, but '#{$3}' is not started with '_'. Please replace '#{$3}' to '_#{$3}' manually."
	elsif line =~ /(\s*)(.*)\s(_[a-zA-Z]\w*)\;\s*\/\/>(.*)/ then
		type = $2
		var_name = $3
		buf = $4.split(",")
		for command in buf do
			command.strip!
			if command=="get" then
				_get = true
			elsif command=="set" then
				_set = true
			end
		end
	end
	return _get,_set,type, var_name
end
def make_get(type,var_name)
	func_name = var_name[1..var_name.size-1]
	type = type.gsub("const","").strip
	if BasicTypes.include?(type) then
		return "#{type} #{func_name}()const{return #{var_name};}"
	else
		return "const #{type}& #{func_name}()const{return #{var_name};}"
	end
end
def make_get(type,var_name)
	func_name = var_name[1..var_name.size-1]
	type = type.gsub("const","").strip
	if BasicTypes.include?(type) then
		return "#{type} #{func_name}()const{return #{var_name};}"
	else
		return "const #{type}& #{func_name}()const{return #{var_name};}"
	end
end

def make_set(type,var_name)
	func_name = var_name[1..var_name.size-1]
	type = type.gsub("const","").strip
	if BasicTypes.include?(type) then
		return "void #{func_name}(#{type} _#{var_name}){#{var_name} = _#{var_name};}"
	else
		return "void #{func_name}(const #{type}& _#{var_name}){#{var_name} = _#{var_name};}"
	end
end


level = 0
current_level = 0
declarations = {}
class_space = ""
File.open(HEADER_FILE).each do |line|
	line =~ /(\s*).*/
	space = $1

	if line =~ /(^|\s+)class\s.*/ then
		class_space = space
	end

	buf = line.strip
	level += buf.count("{")
	level -= buf.count("}")
	if declarations.include?(current_level) and !declarations[current_level].empty? and level < current_level then
		STDOUT.puts "#{class_space}\tprotected:"
		for l,dec in declarations do
			STDOUT.print dec
		end
		declarations[current_level] = []
	elsif line =~ /.*protected:\s+.*/ then
		STDOUT.print line
		for l,dec in declarations do
			STDOUT.print dec
		end
		declarations[current_level] = []
		next
	end
	if current_level - level > 1 then
		for i in 1..current_level-level-1 do
			if declarations.include?(current_level-i) and !declarations[current_level-i].empty? then
				STDERR.puts "ERROR: level get down two or more in a line. This script cannot solve the location for followings."
				STDERR.print declarations[current_level-i]
				STDERR.pust "Please make sure two or more \"}\" should not put in a line."
				exit -1
			end
		end
	end

	current_level = level
	make_get,make_set,var_type,var_name = parse(buf)

	if !make_get && !make_set then
		STDOUT.print line
		next
	end
	# delete command from line
	if declarations.include?(current_level) then
		declarations[current_level] << line.sub(/\/\/>.*/,"")
	else
		declarations[current_level] = line.sub(/\/\/>.*/,"")
	end
	if make_get then
		STDOUT.puts "#{space}#{make_get(var_type,var_name)}"
	end
	if make_set then
		STDOUT.puts "#{space}#{make_set(var_type,var_name)}"
	end

end

