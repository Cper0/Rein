#pragma once

#include<torch/torch.h>

struct SamplingImpl : public torch::nn::Module
{
	SamplingImpl();

	torch::Tensor forward(torch::Tensor mu, torch::Tensor log_var);
};

TORCH_MODULE(Sampling);
