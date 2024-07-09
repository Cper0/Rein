#pragma once

#include<torch/torch.h>
#include<utility>

struct VAEEncoderImpl : torch::nn::Module
{
	VAEEncoderImpl();

	std::pair<torch::Tensor,torch::Tensor> forward(torch::Tensor x);

	torch::nn::Conv2d conv[4] = {nullptr,nullptr,nullptr,nullptr};
	torch::nn::BatchNorm2d norm[4] = {nullptr,nullptr,nullptr,nullptr};
	torch::nn::Dropout drop[4] = {nullptr,nullptr,nullptr,nullptr};

	torch::nn::Linear mu_layer{nullptr};
	torch::nn::Linear log_var_layer{nullptr};
};

TORCH_MODULE(VAEEncoder);
