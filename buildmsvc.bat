@echo off
IF "%VCPKG%"=="" (
	ECHO Environment variable VCPKG is NOT defined, for example SET VCPKG=C:\VCPKG
	EXIT /B
)
IF "%1"=="" ( 
	SET "arch=Arm64"
) ELSE ( 
	SET "arch=%1"
)

SET "generator=Visual Studio 16 2019"

SET "target=%arch%"

IF "%2"=="" ( SET "build=Release" ) ELSE ( SET "build=%2" )
set "builddir=%build%%arch%" 
rem RMDIR .\%builddir% /S /Q
echo Building %build% %arch% ...
cmake -G "%generator%" -A %target% -DCMAKE_BUILD_TYPE=%build% -H. -B%builddir% -DCMAKE_TOOLCHAIN_FILE=%VCPKG%\scripts\buildsystems\vcpkg.cmake -DVCPKG_DIR=%VCPKG%
cmake --build .\%builddir% --config %build%
