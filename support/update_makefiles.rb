#!/usr/bin/ruby

require 'erb'
require 'DependencyList.rb'

TEMPLATE_DIR = "templates"
MOD_MAKEFILE_TEMPLATE = "#{TEMPLATE_DIR}/module_makefile_template.erb"
MOD_SRC_MAKEFILE_TEMPLATE = "#{TEMPLATE_DIR}/module_src_makefile_template.erb"

module_dir = "../modules"
if ARGV.size > 0 then
	module_dir = ARGV[0]
end
MODULE_DIR = module_dir


# check overwriting
if File.exist?("#{MODULE_DIR}/Makefile") then
	STDOUT.print "Makefile already exists. Are you sure to overwrite it?[y/n]: "
	STDOUT.flush
	ans = STDIN.gets.strip
	if ans != "y" then
		exit
	end
end


fout = File.open("#{MODULE_DIR}/Makefile","w")

TEMP = Dir.glob("#{MODULE_DIR}/*")
for temp in TEMP do
	next if !File.directory?(temp)
	temp = File.basename(temp)
	if !ModuleOrder.include?(temp) then
		STDERR.puts "ERROR: unknown dependency for module #{temp}. Edit {skl}/support/DependencyList.rb"
		exit -1
	end
end

MODULE_LIST = Array.new
for mod in ModuleOrder do
	MODULE_LIST << mod
end

STDOUT.puts "Modules: #{MODULE_LIST.join(" ")}"

module_list = Marshal.load(Marshal.dump(MODULE_LIST))
module_list.each{|elem| elem.downcase!}
module_list_d = Marshal.load(Marshal.dump(module_list))
module_list_d.each{|elem| elem + "d"}

fout.puts "default:\n\t$(MAKE) all\n\n"
fout.puts "debug:\n\t$(MAKE) debug_all\n\n"

fout.puts "all: #{module_list.join(" ")}\n\n"
fout.puts "debug_all: #{module_list_d.join(" ")}\n\n"

for mod in MODULE_LIST do
	fout.puts "#{mod.downcase}:\n\tcd #{mod};$(MAKE)"
	fout.puts "#{mod.downcase}d:\n\tcd #{mod};$(MAKE) debug"
end

fout.puts "clean:"
for mod in MODULE_LIST do
	fout.puts "\tcd #{mod};$(MAKE) clean"
end

# generate module MakeFiles
for mod in MODULE_LIST do
	module_makefile = "#{MODULE_DIR}/#{mod}/Makefile"
#	next if File.exist?(module_makefile)
	module_name = mod
	module_downcase_name = mod.downcase

	visual_studio = (VisualStudioProjects.include?(module_name))
	erb = ERB.new(open(MOD_MAKEFILE_TEMPLATE).read)
	File.open(module_makefile,"w").write(erb.result(binding))

	module_src_makefile = "#{MODULE_DIR}/#{mod}/src/Makefile"
	dependency = ModuleDependency[mod].join(" ")
	option_file = "#{TEMPLATE_DIR}/Makefile_#{module_downcase_name}_template.txt"
	if File.exist?(option_file)
		option = File.open(option_file).read
	end
	erb = ERB.new(open(MOD_SRC_MAKEFILE_TEMPLATE).read)
	File.open(module_src_makefile,"w").write(erb.result(binding))
end
