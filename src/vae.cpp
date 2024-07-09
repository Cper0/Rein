#include"vae.hpp"

torch::Tensor VAEImpl::reparameterize(torch::Tensor mu, torch::Tensor log_var)
{
	auto std = torch::exp(0.5 * log_var);
	//auto eps = torch::randn_like(std);
	auto eps = torch::randn_like(std);
	return mu + eps * std;
}

VAEImpl::VAEImpl()
{
	encoder = register_module("encoder", VAEEncoder());
	decoder = register_module("decoder", VAEDecoder());
}

torch::Tensor VAEImpl::forward(torch::Tensor x)
{
	auto [mu, log_var] = encoder->forward(x);
	auto z = reparameterize(mu, log_var);
	return decoder->forward(z);
}
