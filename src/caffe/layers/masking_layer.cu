#include "caffe/layers/masking_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void MaskingLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//caffe_gpu_mul(top[0]->count(), bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), top[0]->mutable_gpu_data()); // multiply mask, y=a*b
		//if (bias_term_) {
		//	caffe_gpu_axpy(top[0]->count(), (Dtype)0.0, this->blobs_[1]->gpu_data(), top[0]->mutable_gpu_data()); // y=a*x+y
		//}
		Forward_cpu(bottom, top);
	}

	template <typename Dtype>
	void MaskingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// TODO: check gradient formulas (http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)

		//// Gradient with respect to bottom data
		//if (propagate_down[0]) {
		//	if (stable_prod_grad_) {
		//		LOG(ERROR) << "stable_prod_grad not yet implemented"; // TODO: see elementwise layer for inspiration
		//	}
		//	else {
		//		caffe_gpu_mul(top[0]->count(), this->blobs_[0]->gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff()); // d_i=w.*d_i+1
		//	}
		//}

		//// Gradient with respect to weights
		//caffe_gpu_mul(top[0]->count(), top[0]->gpu_diff(), bottom[0]->gpu_data(), this->blobs_[0]->mutable_gpu_diff()); // = activation*d_i+1


		//// Gradient with respect to bias
		//if (bias_term_) {
		//	caffe_copy(top[0]->count(), top[0]->gpu_diff(), this->blobs_[1]->mutable_gpu_diff()); // = d_i+1
		//}
		Backward_cpu(top, propagate_down, bottom);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MaskingLayer);

}  // namespace caffe
