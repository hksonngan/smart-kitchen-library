
ModuleDependency = {}
ModuleDependency["Core"] = []
ModuleDependency["OpenCV"] = ["Core"]
ModuleDependency["FlyCapture"] = ["OpenCV","Core"]
ModuleDependency["Kinect"] = ["OpenCV","Core"]

ModuleOrder = []
for mod, dependency in ModuleDependency do
	for dep in dependency do
		if !ModuleOrder.include?(dep) then
			ModuleOrder << dep
		end
	end
	ModuleOrder << mod
end
