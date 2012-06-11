ModuleDependency["Core"] = []
ModuleDependency["OpenCV"] = ["Core"]
ModuleDependency["FlyCapture"] = ["OpenCV","Core"]
ModuleDependency["OpenCVGPU"] = ["OpenCV","Core"]

def sortModule(mod,list)
	# すでに登録されていればtrueを返す
	return true if list.include?(mod)
	# 存在しないモジュール名が入力された場合
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
