#include"encoder.hpp"

EncoderImpl::EncoderImpl()
{
	body = register_module("body", torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::Flatten()
	));

	mu = register_module("mu", torch::nn::Linear(64*64, 2));
	log_var = register_module("log_var", torch::nn::Linear(64*64, 2));
	sampling = register_module("sampling", Sampling());
}

torch::Tensor EncoderImpl::forward(torch::Tensor x)
{
	x = body->forward(x);
	auto m = mu->forward(x);
	auto l = log_var->forward(x);

	return sampling->forward(m, l);
}
