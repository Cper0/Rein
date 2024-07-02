#include"layers.hpp"
#include"util.hpp"

Sigmoid::Sigmoid(size_t input_s) : LayerBase()
{
	input_size = input_s;
}

arma::mat Sigmoid::forward(arma::mat x)
{
	out = util::sigmoid(x);
	return out;
}

arma::mat Sigmoid::backward(arma::mat dy)
{
	const arma::mat d = (1.0 - out) % out;
	return d % dy;
}
	
