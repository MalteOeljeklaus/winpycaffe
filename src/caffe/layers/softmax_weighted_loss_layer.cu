#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_weighted_loss_layer.hpp"
//#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"

// this code has been copied from https://bitbucket.org/deeplab/deeplab-public

namespace caffe {


template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // TODO(Yangqing): implement the GPU version of softmax.
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithWeightedLossLayer);

}  // namespace caffe
