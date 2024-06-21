#include"layers.hpp"
#include"util.hpp"

Sigmoid::Sigmoid(size_t input_s) : LayerBase()
{
	input_size = input_s;
	out = arma::vec(input_size);
}

arma::mat Sigmoid::forward(arma::mat x)
{
	if(x.n_rows != input_size || x.n_cols != 1)
	{
		throw std::logic_error("Error is thrown on forwarding in Sigmoid layer.");
	}

	out = util::sigmoid(x);
	return out;
}

arma::mat Sigmoid::backward(arma::mat dy)
{
	const arma::vec d = (1.0 - out) % out;
	return d % dy;
}
	
