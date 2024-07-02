#include"layers.hpp"

Relu::Relu(size_t input_s) : LayerBase(), mask()
{
	input_size = input_s;
}

arma::mat Relu::forward(arma::mat x)
{
	mask.copy_size(x);
	for(size_t i = 0; i < x.n_rows; i++)
	{
		for(size_t j = 0; j < x.n_cols; j++)
		{
			mask(i,j) = (x(i,j) <= 0.0) ? 0.1 : 1.0;
		}
	}

	return x % mask;
}

arma::mat Relu::backward(arma::mat dy)
{
	return dy % mask;
}
