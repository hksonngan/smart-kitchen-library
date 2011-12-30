#!/usr/bin/env ruby
require 'erb'

class Filter
	attr_reader :name

	def initialize(name,sfiles,hfiles)
		@name=name
		@sfiles=sfiles
		@hfiles=hfiles
	end
	
	def each_src_file(&block)
		@sfiles.each do |f|
			yield f
		end
	end

	def each_header_file(&block)
		@hfiles.each do |f|
			yield f
		end
	end
end

# フィルタに使う接頭辞
names=[
	"Calibration",
	"Exception",
	"ImageAreaSegment",
	"ImageBackgroundImageGenerator",
	"ImageBackgroundSubtract",
	"ImageCamera",
	"Image.*Filter",
	"ImageImage",
	"ParticleFilter"
]

# ファイルをとってくる
sfiles=`ls ../src`.split(/\n/)
hfiles=`ls ../include`.split(/\n/)
# 拡張子のチェック
sfiles.delete_if{|x| not x =~ /(\.c$)|(\.cpp$)/}
hfiles.delete_if{|x| not x =~ /(\.h$)|(\.hpp$)/}

filters=[]

names.each do |name|
	# name に対するフィルタを生成
	r=Regexp.new("^"+name)
	stemp=sfiles.select{|x| x=~r}
	htemp=hfiles.select{|x| x=~r}
	sfiles.delete_if{|x| x=~r}
	hfiles.delete_if{|x| x=~r}
	filters.push Filter.new(name,stemp,htemp)
end

# 残りのやつ
others=Filter.new("def",sfiles,hfiles)

# XML生成
templ=ERB.new(open("OpenCV.tmpl").read)
puts templ.result

