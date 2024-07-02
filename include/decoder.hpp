#pragma once

#include<torch/torch.h>

struct DecoderImpl : public torch::nn::Module
{
	DecoderImpl();

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Linear dense{nullptr};
	torch::nn::Sequential body{nullptr};
};

TORCH_MODULE(Decoder);
