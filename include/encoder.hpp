#pragma once

#include<torch/torch.h>
#include"sampling.hpp"

struct EncoderImpl : public torch::nn::Module
{
	EncoderImpl();

	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d norm1{nullptr};

	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d norm2{nullptr};

	torch::nn::Conv2d conv3{nullptr};
	torch::nn::BatchNorm2d norm3{nullptr};

	torch::nn::Conv2d conv4{nullptr};
	torch::nn::BatchNorm2d norm4{nullptr};

	torch::nn::Linear mu{nullptr};	
	torch::nn::Linear log_var{nullptr};	
	Sampling sampling{nullptr};
};

TORCH_MODULE(Encoder);
