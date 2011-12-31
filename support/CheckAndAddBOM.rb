#!/usr/bin/ruby

if ARGV.size < 1
	STDERR.puts "Usage: #{__FILE__} target.txt"
	exit 0
end

filename = ARGV[0]

if !File.file?(filename) then
	exit 0
end
fin = File.open(filename)
if fin.read(3).unpack("h*").to_s != "febbfb" then
	fin.rewind
	buf = fin.read
	fin.close
	fout = File.open(filename,"w")
	require 'kconv'
	fout.print ["febbfb"].pack("h*")
	fout.print Kconv.toutf8(buf)
end
