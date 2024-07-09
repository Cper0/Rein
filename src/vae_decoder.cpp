#include"vae_decoder.hpp"

VAEDecoderImpl::VAEDecoderImpl()
{
	constexpr double DROP_P = 0.2;

	dense = register_module("dense", torch::nn::Linear(200, 4096));
	dense_norm = register_module("dense_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({4096})));

	conv[0] = register_module("conv0", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 64, 4).stride(2).padding(1)));
	norm[0] = register_module("norm0", torch::nn::BatchNorm2d(64));
	drop[0] = register_module("drop0", torch::nn::Dropout(DROP_P));

	conv[1] = register_module("conv1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 64, 4).stride(2).padding(1)));
	norm[1] = register_module("norm1", torch::nn::BatchNorm2d(64));
	drop[1] = register_module("drop1", torch::nn::Dropout(DROP_P));

	conv[2] = register_module("conv2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1)));
	norm[2] = register_module("norm2", torch::nn::BatchNorm2d(32));
	drop[2] = register_module("drop2", torch::nn::Dropout(DROP_P));

	conv[3] = register_module("conv3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 3, 4).stride(2).padding(1)));
}

torch::Tensor VAEDecoderImpl::forward(torch::Tensor x)
{
	x = dense->forward(x);
	x = dense_norm->forward(x);
	x = torch::leaky_relu(x);
	x = x.view({-1, 64, 8, 8});
	if(torch::isnan(x).any().item<bool>()) throw std::logic_error("b");

	for(int i = 0; i < 3; i++)
	{
		x = conv[i]->forward(x);
		x = norm[i]->forward(x);
		x = torch::leaky_relu(x);
		x = drop[i]->forward(x);
	}

	x = torch::sigmoid(conv[3]->forward(x));
	if(torch::isnan(x).any().item<bool>()) throw std::logic_error("c");
	return x;
}
