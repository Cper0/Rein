#include"layers.hpp"

Relu::Relu(size_t input_s) : LayerBase()
{
	input_size = input_s;
	mask = arma::vec(input_size);
}

arma::mat Relu::forward(arma::mat x)
{
	if(x.n_rows != input_size || x.n_cols != 1)
	{
		std::stringstream st;
		st << "Error is thrown on forwarding in Relu layer.\n";
		throw std::logic_error(st.str());
	}

	for(size_t i = 0; i < x.n_rows; i++) mask[i] = (x[i] <= 0) ? 0 : 1;

	return x % mask;
}

arma::mat Relu::backward(arma::mat dy)
{
	return dy % mask;
}
