import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# init
caffe.set_mode_gpu()
caffe.set_device(0)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('../../data/CamSeq01/0016E5_07959.png')
#im = Image.open('D:/701_StillsRaw_full/0016E5_08125.png')
#im = Image.open('D:/701_StillsRaw_full/Seq05VD_f03360.png')
im = Image.open('D:/701_StillsRaw_full/Seq05VD_f05100.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net

#net = caffe.Net('D:/CIW15_Experimente/2309_IN/4_early_8S_16S/models/bvlc_googlenet/fcn_16stride/fcn-deploy_16stride_early.prototxt', 'F:/2309_OUT/3_early_16s_32s/bin/snapshot/fcn-googlenet-16stride_early_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/CIW15_Experimente/4_late_8s_16s/snapshot/fcn-googlenet-8stride_late_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/2_early_8s_IN/snapshot/fcn-googlenet-8stride_early_iter_42000.caffemodel', caffe.TEST)
net = caffe.Net('D:/CIW15_Experimente/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/4_early_8s_16s/snapshot/fcn-googlenet-8stride_early_iter_14000.caffemodel', caffe.TEST)

#net = caffe.Net('D:/CIW15_Experimente/2209_OUT/2_32s_IN/models/bvlc_googlenet/fcn_32stride/fcn-deploy_32stride.prototxt', 'D:/CIW15_Experimente/2209_OUT/2_32s_IN/bin/snapshot/fcn-googlenet-32stride_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/2209_OUT/2_early_8s_IN/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/2209_OUT/2_early_8s_IN/bin/snapshot/fcn-googlenet-8stride_early_iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('D:/CIW15_Experimente/2209_OUT/2_late_8s_IN/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/CIW15_Experimente/2209_OUT/2_late_8s_IN/bin/snapshot/fcn-googlenet-8stride_late_iter_14000.caffemodel', caffe.TEST)


#net = caffe.Net('D:/Stick/2_32s_IN/models/bvlc_googlenet/fcn_32stride/fcn-deploy_32stride.prototxt', 'D:/Stick/2_32s_IN/bin/snapshot/fcn-googlenet-32stride_iter_1.caffemodel', caffe.TEST)
#net = caffe.Net('D:/Stick/2_early_8s_IN/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/Stick/2_early_8s_IN/bin/snapshot/fcn-googlenet-8stride_early_iter_1.caffemodel', caffe.TEST)
#net = caffe.Net('D:/Stick/2_late_8s_IN/models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt', 'D:/Stick/2_late_8s_IN/bin/snapshot/fcn-googlenet-8stride_late_iter_1.caffemodel', caffe.TEST)


#net = caffe.Net('D:/Stick/1_32s_IN/models/bvlc_googlenet/fcn_32stride/fcn-deploy_32stride.prototxt', 'D:/Stick/1_32s_IN/bin/snapshot/fcn-train_val_32stride__iter_1.caffemodel', caffe.TEST)
#net = caffe.Net('../../models/bvlc_googlenet/fcn_32stride/fcn-deploy_32stride.prototxt', 'D:/CIW15_Experimente/1609/32/snapshot/fcn-googlenet-32stride__iter_14000.caffemodel', caffe.TEST)
#net = caffe.Net('../../models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt', 'D:/CIW15_Experimente/2109_out/1_early_8s_IN/bin/snapshot/fcn-googlenet-8stride_early__iter_75000.caffemodel', caffe.TEST)
#net = caffe.Net('../../models/bvlc_googlenet/fcn_16stride/fcn-deploy_16stride_early.prototxt','D:/CIW15_Experimente/2109_out/1_early_16s_IN/bin/snapshot/fcn-train_val_16stride_early__iter_17000.caffemodel',caffe.TEST)
#net = caffe.Net('../../models/bvlc_googlenet/fcn_16stride/fcn-deploy_16stride_early.prototxt','D:\\Stick\\1_early_16s_IN\\bin\\snapshot\\fcn-googlenet-16stride_init_iter_14000.caffemodel',caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
assert isinstance(net, object)
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
out[1,1] = 11
#print out

#train_loss = np.load('D:/camvid_experiment_07/loss-googlenet8-early_camvid_finetune_iter_6000.npy')
#train_loss1 = np.load('D:/camvid_experiment_07/loss-googlenet8_late-bvlc_camvid_finetune_iter_7000.npy')

#plt.subplot(1, 2, 1)
#plt.imshow(im)
#plt.subplot(1, 2, 2)
plt.imshow(out)
#plt.subplot(1, 3, 3)
#plt.plot(moving_average(train_loss[0:6001],15))
#plt.plot(moving_average(train_loss1[0:7001],15))

#plt.show()