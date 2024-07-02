#include"decoder.hpp"

DecoderImpl::DecoderImpl()
{
	dense = register_module("rectify", torch::nn::Linear(2, 64*64*64));
	body = register_module("body", torch::nn::Sequential(
		torch::nn::Unflatten(torch::nn::UnflattenOptions(1, {64, 64, 64})),
		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 64, 3).stride(1).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 64, 4).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 3, 4).stride(2).padding(1)),
		torch::nn::LeakyReLU(),
		torch::nn::Sigmoid()
	));
}

torch::Tensor DecoderImpl::forward(torch::Tensor x)
{
	x = dense->forward(x);
	x = body->forward(x);
	return x;
}
