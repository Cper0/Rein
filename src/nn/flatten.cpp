#include"layers.hpp"

Flatten::Flatten(size_t in_rows, size_t in_cols) : LayerBase()
{
	input_rows = in_rows;
	input_cols = in_cols;
}

arma::mat Flatten::forward(arma::mat x)
{
	if(x.n_rows != input_rows || x.n_cols != input_cols)
	{
		throw std::logic_error("Errror is thrown in Flatten layer.");
	}
	return arma::vectorise(x);
}

arma::mat Flatten::backward(arma::mat dy)
{
	dy.reshape(input_rows, input_cols);
	return dy.t();
}
