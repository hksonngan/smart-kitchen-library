#!/usr/bin/ruby -Ku

SRC_EXT = [".cpp",".c",".hpp",".cu"]
HEADER_EXT = [".h",".hpp",".inc"]

require 'optparse'
require 'kconv'

def getRelativePath(base,target)
	# this code is from "http://blade.nagaokaut.ac.jp/cgi-bin/scat.rb/ruby/ruby-list/36985"
	sep = /#{File::SEPARATOR}+/o
		base = base.split(sep)
	base.pop
	target = target.split(sep)
	while base.first == target.first
		base.shift
		target.shift
	end
	return File.join([".."]*base.size+target)
end

def getFileList(project_file,directories,filter)
	abs_proj_file = File.expand_path(project_file)
	list = []
	for dir in directories do
		rel_path = getRelativePath(abs_proj_file,File.expand_path(dir))
		for file in Dir.glob("#{dir}/**") do
			ext = File.extname(file)
			next if ext.empty?()
			next unless filter.include?(ext)
			list << "#{rel_path}/#{File.basename(file)}".gsub("/","\\")
		end
	end
	return list
end


opt = OptionParser.new

input_file = ""
opt.on('-i FILE', '--input', "input project file of VisualC++"){|v| input_file = v}

# set src file directories
src_file_directories = ["../src/"]
opt.on('-s <DIR:DIR:...>','--src_dir',"directories whose files with proper extention you want to add to the project as src file."){|v| 
	if !v.strip.empty?() then
		src_file_directories.clear
		v.split(":").each{|dir|
			src_file_directories << dir.strip
		}
	end
}

# set header file directories
header_file_directories = ["../include/"]
opt.on('-h <DIR:DIR:...>','--header_dir', "directories whose files with proper extension you want to add to the project as header file."){|v|
	if !v.strip.empty?() then
		header_file_directories.clear
		v.split(":").each{|dir|
			header_file_directories << dir.strip
		}
	end
}

opt.parse!(ARGV)

unless File.exist?(input_file) then
	STDERR.puts "Error: '#{input_file}' does not exist."
	exit -1
end

source = File.open(input_file).read

def renewList(buf, tag, project_file,directories,filter)
	exist_files = getFileList(project_file,directories,filter)
	registered_files = []
	temp = buf
	while /<#{tag} Include=\"(.*?)\"\s*\/>([\W\w]*)/ =~ temp do
		registered_files << $1
		temp = $2
	end
	registered_files.sort!
	exist_files.sort!
	temp = buf.clone
	added = exist_files - registered_files
	deleted = registered_files - exist_files
#	puts "Deleted"
#	puts deleted
#	puts "Added"
#	puts added
	for file in deleted do
		regex = /(\n*\s*<#{tag} Include=\"#{file.gsub('\\','\\\\\\\\')}\"\s*\/>)(\s*)/
		if Regexp.compile(regex) =~ temp then
			temp.sub!($1+$2,$2)
		else
			STDERR.puts "Error failed to find the entry for '#{file}.'"
			exit 0
		end
	end
	for file in added do
		temp += "\n    <#{tag} Include=\"#{file}\"/>"
	end

	return temp
end

# ソースファイルが書かれたItemGroupを取得
result = source.clone
src_list = []
#while /.*<ClCompile Include='(\.\.\\src\\.+\.cpp)'\/>(.*)<\/ItemGroup>.*/ =~ source do
while /<ItemGroup>([\w\W]*?)<\/ItemGroup>([\w\W]*)/ =~ source do
	buf = $1
	source = $2
	if /<ClCompile Include="..\\src\\.+\.(c|cpp|cu)"\s*\/>/ =~ buf then
#    <ClCompile Include="..\src\BackgroundSubtractAlgorithm.cpp" />
		new_buf = renewList(buf,"ClCompile",input_file,src_file_directories,SRC_EXT)
		result.sub!(buf,new_buf)
#		p "source file updated"
	elsif /<ClInclude Include=\"..\\include\\.+\.(h|hpp)\"\s*\/>/ =~ buf then
		new_buf = renewList(buf,"ClInclude",input_file,header_file_directories, HEADER_EXT)
		result.sub!(buf,new_buf)
#		p "header file updated"
	end
end

print result


