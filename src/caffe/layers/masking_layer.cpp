#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/masking_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MaskingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		bias_term_ = false; // TODO: read from proto

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			// initialize weights (and bias), adjust parameter blob shape(s) and fill the values

			if (bias_term_) {
				this->blobs_.resize(2);
				this->blobs_[1].reset(new Blob<Dtype>(bottom[0]->shape()));
			}
			else {
				this->blobs_.resize(1);
			}
			this->blobs_[0].reset(new Blob<Dtype>(bottom[0]->shape()));

			if (bias_term_) {
				FillerParameter bias_filler_param;	// TODO: read this from prototxt
				bias_filler_param.set_type("constant");
				bias_filler_param.set_value(0.0);
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
				bias_filler->Fill(this->blobs_[1].get());
			}

			FillerParameter weight_filler_param;	// TODO: read this from prototxt
			weight_filler_param.set_type("constant");
			weight_filler_param.set_value(1.0);

			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(weight_filler_param));
			weight_filler->Fill(this->blobs_[0].get());
		}

		stable_prod_grad_ = true; // TODO: read this from proto = this->layer_param_.masking_param().stable_prod_grad();
	}

	template <typename Dtype>
	void MaskingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// TODO: add checks

		if (bias_term_) {
			this->blobs_[1]->Reshape(bottom[0]->shape());
		}
		this->blobs_[0]->Reshape(bottom[0]->shape());
		top[0]->Reshape(bottom[0]->shape());
	}

	template <typename Dtype>
	void MaskingLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		caffe_mul(top[0]->count(), bottom[0]->cpu_data(), this->blobs_[0]->cpu_data(), top[0]->mutable_cpu_data()); // multiply mask, y=a*b
		if (bias_term_) {
			caffe_axpy(top[0]->count(), (Dtype)0.0, this->blobs_[1]->cpu_data(), top[0]->mutable_cpu_data()); // y=a*x+y
		}
	}

	template <typename Dtype>
	void MaskingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		CHECK_GE(this->blobs_.size(), 1);

		// TODO: check gradient formulas (http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)

		if (stable_prod_grad_) {
			if (propagate_down[0]) {
				// Gradient with respect to bottom data
				caffe_mul(top[0]->count(), this->blobs_[0]->cpu_data(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff()); // d_i = d_(i+1) .* w
			}

			// Gradient with respect to weights
			caffe_mul(top[0]->count(), bottom[0]->cpu_data(), top[0]->cpu_diff(), this->blobs_[0]->mutable_cpu_diff()); // d_i = d_(i+1) .* in

			// Gradient with respect to bias
			if (bias_term_) {
				LOG(ERROR) << "bias gradient not yet implemented"; // TODO: see elementwise layer for inspiration
				//caffe_copy(top[0]->count(), top[0]->cpu_diff(), this->blobs_[1]->mutable_cpu_diff()); // = d_i+1
			}
		} else {
			// less stable gradient computation method inspired by elementwise layer, this is just for comparison/debugging purposes

			if (propagate_down[0]) {
				// Gradient with respect to bottom data
				caffe_div(top[0]->count(), top[0]->cpu_data(), bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff());
				caffe_mul(top[0]->count(), bottom[0]->mutable_cpu_diff(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
			}

			// Gradient with respect to weights
			caffe_div(top[0]->count(), top[0]->cpu_data(), blobs_[0]->cpu_data(), this->blobs_[0]->mutable_cpu_diff());
			caffe_mul(top[0]->count(), this->blobs_[0]->mutable_cpu_diff(), top[0]->cpu_diff(), this->blobs_[0]->mutable_cpu_diff());

			// Gradient with respect to bias
			if (bias_term_) {
				LOG(ERROR) << "unstable bias gradient not yet implemented"; // TODO: see elementwise layer for formulas
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MaskingLayer);
#endif

	INSTANTIATE_CLASS(MaskingLayer);
	REGISTER_LAYER_CLASS(Masking);

}  // namespace caffe
