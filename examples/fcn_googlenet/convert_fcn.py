# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import os
os.environ['GLOG_minloglevel'] = '1'


import caffe
import numpy as np
import inspect

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = np.float32((size + 1) // 2)
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            print l            
            print m
            print k
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
#        net.params[l][0].data[range(m), range(k), :, :] = filt
        global local_interp_surgery
        local_interp_surgery = inspect.currentframe().f_locals



def convert_fcn(original_deploy, fcn_deploy, original_params, fcn_params, original_model_file, fcn_model_file_output):
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(original_deploy, original_model_file, caffe.TEST)
    
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(fcn_deploy, original_model_file, caffe.TEST)

    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in original_params}
 
    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in fcn_params}
    
    for pr, pr_conv in zip(original_params, fcn_params):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in net_full_conv.params.keys() if 'up' in k]
    interp_surgery(net_full_conv, interp_layers) # or use bilinear upsampling weight filler instead
    
    net_full_conv.save(fcn_model_file_output)
    global local_convert_fcn
    local_convert_fcn = inspect.currentframe().f_locals

convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/fcn_googlenet/fcn-deploy_8stride_late_cc3_camvid.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/fcn_googlenet/init-fcn-deploy_8stride_late_cc3_camvid.caffemodel')