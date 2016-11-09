# experimental branch of caffe (see below) with the following modifications:
- PR #2016 for reduced memory usage
- added SoftmaxWithWeightedLossLayer from DeepLab https://bitbucket.org/deeplab/deeplab-public/
- added experimental MaskingLayer
- make.bat assumes MSVC2015, CUDA installed and cudnn-8.0-windows10-x64-v5.1.zip present in the caffe root directory

# References

http://caffe.berkeleyvision.org

https://github.com/BVLC/caffe/tree/windows