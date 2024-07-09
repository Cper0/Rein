#pragma once

#include<torch/torch.h>

struct VAEDecoderImpl : torch::nn::Module
{
	VAEDecoderImpl();

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Linear dense = nullptr;
	torch::nn::LayerNorm dense_norm = nullptr;

	torch::nn::ConvTranspose2d conv[4] = {nullptr,nullptr,nullptr,nullptr};
	torch::nn::BatchNorm2d norm[3] = {nullptr,nullptr,nullptr};
	torch::nn::Dropout drop[3] = {nullptr,nullptr,nullptr};
};

TORCH_MODULE(VAEDecoder);
