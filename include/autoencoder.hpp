#pragma once

#include"encoder.hpp"
#include"decoder.hpp"
#include<torch/torch.h>

struct AutoEncoder : public torch::nn::Module
{
	AutoEncoder();

	torch::Tensor forward(torch::Tensor x);

	Encoder encoder{nullptr};
	Decoder decoder{nullptr};
};
