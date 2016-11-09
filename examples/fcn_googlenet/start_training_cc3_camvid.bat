@echo off
mkdir logs
mkdir snapshot
set GLOG_log_dir=%cd%\logs
..\..\bin\caffe train -solver ..\..\models\fcn_googlenet\fcn-solver_cc3_camvid.prototxt -weights ..\..\models\fcn_googlenet\init-fcn-deploy_8stride_late_cc3_camvid.caffemodel