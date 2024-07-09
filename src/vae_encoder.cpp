#include"vae_encoder.hpp"

VAEEncoderImpl::VAEEncoderImpl()
{
	constexpr double DROP_P = 0.2;

	conv[0] = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(2).padding(1)));
	norm[0] = register_module("norm1", torch::nn::BatchNorm2d(32));
	drop[0] = register_module("drop1", torch::nn::Dropout(DROP_P));

	conv[1] = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
	norm[1] = register_module("norm2", torch::nn::BatchNorm2d(64));
	drop[1] = register_module("drop2", torch::nn::Dropout(DROP_P));

	conv[2] = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(2).padding(1)));
	norm[2] = register_module("norm3", torch::nn::BatchNorm2d(64));
	drop[2] = register_module("drop3", torch::nn::Dropout(DROP_P));

	conv[3] = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(2).padding(1)));
	norm[3] = register_module("norm4", torch::nn::BatchNorm2d(64));
	drop[3] = register_module("drop4", torch::nn::Dropout(DROP_P));

	mu_layer = register_module("mu", torch::nn::Linear(4096, 200));
	log_var_layer = register_module("log_var", torch::nn::Linear(4096, 200));
}

std::pair<torch::Tensor,torch::Tensor> VAEEncoderImpl::forward(torch::Tensor x)
{
	x = x.view({-1, 3, 128, 128});

	for(int i = 0; i < 4; i++)
	{
		x = conv[i]->forward(x);
		x = norm[i]->forward(x);
		x = torch::leaky_relu(x);
		x = drop[i]->forward(x);
	}

	x = x.view({-1, 4096});
	if(torch::isnan(x).any().item<bool>()) throw std::logic_error("a");

	auto mu = mu_layer->forward(x);
	auto log_var = log_var_layer->forward(x);

	return std::make_pair(mu.clone(), log_var.clone());
}
