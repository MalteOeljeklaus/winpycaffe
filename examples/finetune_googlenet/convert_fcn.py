# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
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
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


def convert_fcn(original_deploy, fcn_deploy, original_params, fcn_params, original_model_file, fcn_model_file_output):
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(original_deploy, original_model_file, caffe.TEST)
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in original_params}
    
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(fcn_deploy, original_model_file, caffe.TEST)
    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in fcn_params}
    
    for pr, pr_conv in zip(original_params, fcn_params):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in net_full_conv.params.keys() if 'up' in k]
    interp_surgery(net_full_conv, interp_layers)
    
    net_full_conv.save(fcn_model_file_output)



convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_early.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/bvlc_googlenet/fcn_8stride/fcn-googlenet_8stride_early.caffemodel')

convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/bvlc_googlenet/fcn_8stride/fcn-deploy_8stride_late.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/bvlc_googlenet/fcn_8stride/fcn-googlenet_8stride_late.caffemodel')

convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/bvlc_googlenet/fcn_16stride/fcn-deploy_16stride_early.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/bvlc_googlenet/fcn_16stride/fcn-googlenet_16stride_early.caffemodel')

convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/bvlc_googlenet/fcn_16stride/fcn-deploy_16stride_late.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/bvlc_googlenet/fcn_16stride/fcn-googlenet_16stride_late.caffemodel')

convert_fcn('../../models/bvlc_googlenet/deploy.prototxt','../../models/bvlc_googlenet/fcn_32stride/fcn-deploy_32stride.prototxt',
            ['loss3/classifier'],['loss3/classifier-conv'],
            '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel','../../models/bvlc_googlenet/fcn_32stride/fcn-googlenet_32stride.caffemodel')
