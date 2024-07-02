#include"autoencoder.hpp"

AutoEncoder::AutoEncoder()
{
	encoder = register_module("encoder", Encoder());
	decoder = register_module("decoder", Decoder());
}

torch::Tensor AutoEncoder::forward(torch::Tensor x)
{
	x = encoder->forward(x);
	x = decoder->forward(x);
	return x;
}
