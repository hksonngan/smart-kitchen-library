#VSPROJ = VisualStudioProject/SKLCore.vcxproj

all:
	cd src;$(MAKE)
	cp src/libskl_core.a ../lib
#	ruby ../support/VSProjectFileMaintainer.rb -i $(VSPROJ) -s src -h include > temp.proj;mv temp.proj $(VSPROJ)

.PHONY: clean debug

check-syntax:
	g++ -o null -Wall -S ${CHK_SOURCES}

debug:
	cd src;$(MAKE) debug
	cp src/libskl_core_d.a ../lib

clean:
	cd src;$(MAKE) clean
