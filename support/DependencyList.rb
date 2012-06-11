ModuleDependency["Core"] = []
ModuleDependency["OpenCV"] = ["Core"]
ModuleDependency["FlyCapture"] = ["OpenCV","Core"]
ModuleDependency["OpenCVGPU"] = ["OpenCV","Core"]

def sortModule(mod,list)
	# $B$9$G$KEPO?$5$l$F$$$l$P(Btrue$B$rJV$9(B
	return true if list.include?(mod)
	# $BB8:_$7$J$$%b%8%e!<%kL>$,F~NO$5$l$?>l9g(B
	return false unless ModuleDependency.include?(mod)

	for dep in ModuleDependency[mod] do
		return false unless sortModule(dep,list)
	end
	list << mod
	return true
end

ModuleOrder = []
for mod, dependency in ModuleDependency do
	if !sortModule(mod,ModuleOrder) then
		STDERR.puts "ERROR: failed to make ModuleOrderList."
	end
end
ModuleOrder.reverse!
