#pragma once

#include<torch/torch.h>

#include"agent_actions.hpp"

struct ActionEncoder : torch::nn::Module
{
	ActionEncoder();

	torch::Tensor forward(torch::Tensor x);

	torch::nn::Linear dense[4] = {nullptr,nullptr,nullptr,nullptr};
	torch::nn::LayerNorm norm[4] = {nullptr,nullptr,nullptr,nullptr};
};

