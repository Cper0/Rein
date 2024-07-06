#include"encoder.hpp"

EncoderImpl::EncoderImpl()
{
	conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(2).padding(1)));
	norm1 = register_module("norm1", torch::nn::BatchNorm2d(32));

	conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
	norm2 = register_module("norm2", torch::nn::BatchNorm2d(64));

	conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(2).padding(1)));
	norm3 = register_module("norm3", torch::nn::BatchNorm2d(64));

	conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
	norm4 = register_module("norm4", torch::nn::BatchNorm2d(64));

	mu = register_module("mu", torch::nn::Linear(64*64*64, 2));
	log_var = register_module("log_var", torch::nn::Linear(64*64*64, 2));
	sampling = register_module("sampling", Sampling());
}

torch::Tensor EncoderImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = torch::leaky_relu(norm1->forward(x));

	x = conv2->forward(x);
	x = torch::leaky_relu(norm2->forward(x));

	x = conv3->forward(x);
	x = torch::leaky_relu(norm3->forward(x));

	x = conv4->forward(x);
	x = torch::leaky_relu(norm4->forward(x));

	x = x.view({-1, 64 * 64 * 64});

	auto m = mu->forward(x);
	auto l = log_var->forward(x);

	return sampling->forward(m, l);
}
