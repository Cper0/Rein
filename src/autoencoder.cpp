#include"autoencoder.hpp"

AutoEncoder::AutoEncoder()
{
	encoder = register_module("encoder", Encoder());
	decoder = register_module("decoder", Decoder());
}

torch::Tensor AutoEncoder::forward(torch::Tensor x)
{
	encoder->to(torch::kCUDA);
	decoder->to(torch::kCUDA);
	x = encoder->forward(x);
	x = decoder->forward(x);
	return x;
}
