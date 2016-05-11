#include "caffe/layers/masking_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void MaskingLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		caffe_gpu_mul(top[0]->count(), bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), top[0]->mutable_gpu_data()); // multiply mask, y=a*b
		if (bias_term_) {
			caffe_gpu_axpy(top[0]->count(), (Dtype)0.0, this->blobs_[1]->gpu_data(), top[0]->mutable_gpu_data()); // y=a*x+y
		}
	}

	template <typename Dtype>
	void MaskingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		CHECK_GE(this->blobs_.size(), 1);

		// TODO: check gradient formulas (http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)

		if (stable_prod_grad_) {
			if (propagate_down[0]) {
				// Gradient with respect to bottom data
				caffe_gpu_mul(top[0]->count(), this->blobs_[0]->gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff()); // d_i = d_(i+1) .* w
			}

			// Gradient with respect to weights
			caffe_gpu_mul(top[0]->count(), bottom[0]->gpu_data(), top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff()); // d_i = d_(i+1) .* in

			// Gradient with respect to bias
			if (bias_term_) {
				LOG(ERROR) << "bias gradient not yet implemented"; // TODO: see elementwise layer for inspiration
				//caffe_copy(top[0]->count(), top[0]->cpu_diff(), this->blobs_[1]->mutable_cpu_diff()); // = d_i+1
			}
		}
		else {
			// less stable gradient computation method inspired by elementwise layer, this is just for comparison/debugging purposes

			if (propagate_down[0]) {
				// Gradient with respect to bottom data
				caffe_gpu_div(top[0]->count(), top[0]->gpu_data(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
				caffe_gpu_mul(top[0]->count(), bottom[0]->gpu_diff(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
			}

			// Gradient with respect to weights
			caffe_gpu_div(top[0]->count(), top[0]->gpu_data(), blobs_[0]->gpu_data(), this->blobs_[0]->mutable_gpu_diff());
			caffe_gpu_mul(top[0]->count(), this->blobs_[0]->gpu_diff(), top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff());

			// Gradient with respect to bias
			if (bias_term_) {
				LOG(ERROR) << "unstable bias gradient not yet implemented"; // TODO: see elementwise layer for formulas
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MaskingLayer);

}  // namespace caffe
