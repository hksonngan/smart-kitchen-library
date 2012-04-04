#!/usr/bin/ruby -Ku

SRC_EXT = [".cpp",".c",".hpp",".cu"]
HEADER_EXT = [".h",".hpp",".inc"]

require 'rexml/document'
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

source = File.open(input_file)
doc = REXML::Document.new source.read

# ソースファイルとヘッダファイルのフィルタを取得
src_filter = []
header_filter = []

doc.each_element("/Project/ItemGroup"){|node|
	next if node.attribute("Label")!=nil

	if node.elements["./ClCompile"]!=nil then
		# ソースファイルを格納するItemGroup
		exist_file_list = []
		node.elements.each("./ClCompile"){|elem|
			exist_file_list << elem.attribute('Include').to_s.strip
		}
		file_list = getFileList(input_file, src_file_directories, SRC_EXT)
		for file in file_list do
			file.strip!
			# 既にプロジェクトに含まれているファイルは無視
			if exist_file_list.include?(file) then
				exist_file_list.delete(file)
				next
			end
			attr = {"Include" => file}
			node.add_element("ClCompile",attr)
		end
	elsif node.elements["./ClInclude"]!=nil then
		# ヘッダファイルを格納するItemGroup
		exist_file_list = []
		node.elements.each("./ClInclude"){|elem|
			exist_file_list << elem.attribute('Include').to_s.strip
		}
		file_list = getFileList(input_file, header_file_directories, HEADER_EXT)
		for file in file_list do
			# 既にプロジェクトに含まれているファイルは無視
			if exist_file_list.include?(file) then
#				STDERR.puts file
				exist_file_list.delete(file)
				next
			end
			attr = {"Include" => file}
			node.add_element("ClInclude",attr)
		end
	end
}

doc.write STDOUT

source.close
