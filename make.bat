if exist ".\libraries" (
	echo found 3rdparty dependencies, all good!
) else (
	echo 3rdparty dependencies not found, downloading...
	python scripts\download_prebuilt_dependencies.py --msvc_version=v140
	del libraries_v140_x64_py27_*.tar.bz2
)

ECHO REM Set WshShell = CreateObject("Wscript.Shell") > j_unzip.vbs 
ECHO REM CurDir = WshShell.ExpandEnvironmentStrings("%%cd%%") >> j_unzip.vbs 
ECHO Dim sCurPath >> j_unzip.vbs 
ECHO sCurPath = CreateObject("Scripting.FileSystemObject").GetAbsolutePathName(".") >> j_unzip.vbs 
ECHO strZipFile = sCurPath ^& "\" ^& "cudnn-8.0-windows10-x64-v5.1.zip" >> j_unzip.vbs 
ECHO Set objShell = CreateObject( "Shell.Application" ) >> j_unzip.vbs 
ECHO Set objSource = objShell.NameSpace(strZipFile).Items() >> j_unzip.vbs 
ECHO Set objTarget = objShell.NameSpace(sCurPath ^& "\libraries\cudnn\") >> j_unzip.vbs 
ECHO intOptions = 256 >> j_unzip.vbs 
ECHO objTarget.CopyHere objSource, intOptions >> j_unzip.vbs

if exist ".\libraries\cudnn" (
	echo found cudnn dependencies, all good!
) else (
	echo searching for cudnn dependencies...
	if exist ".\cudnn-8.0-windows10-x64-v5.1.zip" (
		echo found cudnn archive, extracting...
		mkdir libraries\cudnn
		cscript /B j_unzip.vbs
		del cudnn-8.0-windows10-x64-v5.1.zip
		copy "%CUDA_PATH_V8_0%\bin\cublas64_80.dll" libraries\bin
		copy "%CUDA_PATH_V8_0%\bin\cudart64_80.dll" libraries\bin
		copy "%CUDA_PATH_V8_0%\bin\curand64_80.dll" libraries\bin
	) else (
		echo ### Please download cudnn-8.0-windows10-x64-v5.1.zip to your caffe root folder to proceed ###
		del j_unzip.vbs
		pause>nul
		exit -1
	)
)

del j_unzip.vbs

call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
set CAFFE_DEPENDENCIES=..\libraries
set CMAKE_CONFIGURATION=Release
mkdir build
cd build
"C:\Program Files\CMake\bin\cmake.exe" -G%CMAKE_GENERATOR% -DBLAS=Open -DCMAKE_BUILD_TYPE=%CMAKE_CONFIGURATION% -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=^<install_path^> -DCUDNNROOT%CAFFE_DEPENDENCIES%\cudnn= -C %CAFFE_DEPENDENCIES%\caffe-builder-config.cmake  ..\
"C:\Program Files\CMake\bin\cmake.exe" --build . --config %CMAKE_CONFIGURATION%
"C:\Program Files\CMake\bin\cmake.exe" --build . --config %CMAKE_CONFIGURATION% --target install