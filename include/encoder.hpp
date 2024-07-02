#pragma once

#include<torch/torch.h>
#include"sampling.hpp"

struct EncoderImpl : public torch::nn::Module
{
	EncoderImpl();

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Sequential body{nullptr};
	torch::nn::Linear mu{nullptr};	
	torch::nn::Linear log_var{nullptr};	
	Sampling sampling{nullptr};
};

TORCH_MODULE(Encoder);
