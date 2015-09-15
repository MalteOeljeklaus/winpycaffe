set thirdpartyurl=https://www.dropbox.com/s/xtcodx7i5zlukza/3rdparty_caffe_27082015_win64.7z?dl=1

if exist ".\3rdparty" (
	echo found 3rdparty dependencies, all good!
) else (
	echo 3rdparty dependencies not found...
	if exist ".\cudnn-7.0-win-x64-v3.0-prod.zip" (
		call .\tools\wget_1.11.4_cygwin\wget.exe --no-check-cert %thirdpartyurl% -O 3rdparty_tmp.7z
		call .\tools\7z938-extra\7za.exe x -y 3rdparty_tmp.7z
		del 3rdparty_tmp.7z
		call .\tools\7z938-extra\7za.exe e -y cudnn-7.0-win-x64-v3.0-prod.zip -o3rdparty\lib cuda\lib\x64
		call .\tools\7z938-extra\7za.exe e -y cudnn-7.0-win-x64-v3.0-prod.zip -o3rdparty\bin cuda\bin
		call .\tools\7z938-extra\7za.exe e -y cudnn-7.0-win-x64-v3.0-prod.zip -o3rdparty\include cuda\include
		del cudnn-7.0-win-x64-v3.0-prod.zip
		copy "%CUDA_PATH_V7_5%\bin\cublas64_75.dll" 3rdparty\bin
		copy "%CUDA_PATH_V7_5%\bin\cudart64_75.dll" 3rdparty\bin
		copy "%CUDA_PATH_V7_5%\bin\curand64_75.dll" 3rdparty\bin

	) else (
		echo ### Please download cudnn-7.0-win-x64-v3.0-prod.zip to your caffe root folder to proceed ###
		exit -1
	)
)

if exist "./src/caffe/proto/caffe.pb.h" (
    echo caffe.pb.h remains the same as before
) else (
    echo caffe.pb.h is being generated
    "./tools/protoc" -I="./src/caffe/proto" --cpp_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"
)

if exist "./src/caffe/proto/*.py" (
    echo caffe python proto definitions remain the same as before
) else (
    echo caffe python proto definitions are being generated
    "./tools/protoc" -I="./src/caffe/proto" --python_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"
)