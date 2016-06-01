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
			LOG(INFO) << "top[0]->count()=" << std::to_string(top[0]->count()) << ", propagate_down[0]=" << std::to_string(propagate_down[0]) << ", bottom[0]->mutable_gpu_diff()=" << std::to_string((unsigned int)(bottom[0]->mutable_gpu_diff())) << ", top[0]->gpu_diff()=" << std::to_string((unsigned int)(top[0]->gpu_diff()));
			if (propagate_down[0]) {
				// Gradient with respect to bottom data
				caffe_gpu_mul(top[0]->count(), this->blobs_[0]->gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff()); // d_i = d_(i+1) .* w
			}

			// Gradient with respect to weights
			LOG(INFO) << "top[0]->count()=" << std::to_string(top[0]->count()) << ", propagate_down[1]=" << std::to_string(propagate_down[1]) << ", blobs_[0]->mutable_gpu_diff()=" << std::to_string((unsigned int)(blobs_[0]->mutable_gpu_diff())) << ", top[0]->gpu_diff()=" << std::to_string((unsigned int)(top[0]->gpu_diff()));
			caffe_gpu_mul(top[0]->count(), bottom[0]->gpu_data(), top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff()); // d_i = d_(i+1) .* in
//			caffe_copy(top[0]->count(), top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff()); // d_i = d_(i+1)

			// Gradient with respect to bias
			if (bias_term_) {
				// LOG(ERROR) << "bias gradient not yet implemented"; // TODO: see elementwise layer for inspiration
				LOG(INFO) << "top[0]->count()=" << std::to_string(top[0]->count()) << ", propagate_down[2]=" << std::to_string(propagate_down[2]) << ", blobs_[1]->mutable_gpu_diff()=" << std::to_string((unsigned int)(blobs_[1]->mutable_gpu_diff())) << ", top[0]->gpu_diff()=" << std::to_string((unsigned int)(top[0]->gpu_diff()));
				caffe_copy(top[0]->count(), top[0]->gpu_diff(), this->blobs_[1]->mutable_gpu_diff()); // d_i = d_(i+1)
//				caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(), this->blobs_[0]->mutable_gpu_data());
			}
		}
		else {
			// less stable gradient computation method inspired by elementwise layer, this is just for comparison/debugging purposes
			// TODO: this is erroneous if bias is used

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
				// LOG(ERROR) << "unstable bias gradient not yet implemented"; // TODO: see elementwise layer for formulas
				caffe_copy(top[0]->count(), top[0]->cpu_diff(), this->blobs_[1]->mutable_cpu_diff()); // d_i = d_(i+1)
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MaskingLayer);

}  // namespace caffe
