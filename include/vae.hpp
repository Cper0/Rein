#pragma once

#include<torch/torch.h>

#include"vae_encoder.hpp"
#include"vae_decoder.hpp"

struct VAEImpl : torch::nn::Module
{
	static torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

	VAEImpl();

	torch::Tensor forward(torch::Tensor x);

	VAEEncoder encoder;
	VAEDecoder decoder;
};

TORCH_MODULE(VAE);
