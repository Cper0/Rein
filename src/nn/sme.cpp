#include"layers.hpp"

SME::SME(size_t input_s) : in_x(), in_t()
{
	input_size = input_s;
}

double SME::forward(arma::vec x, arma::vec t)
{
	if(x.n_elem != input_size || t.n_elem != input_size)
	{
		throw std::logic_error("Error is thrown on forwarding in SME layer.");
	}

	in_x = x;
	in_t = t;
	
	const double s = arma::sum((x - t) % (x - t));
	const double m = s / x.n_elem;
	return m;
}

arma::vec SME::backward(double dy)
{
	const arma::vec dx = (in_x - in_t) / input_size * 2.0;
	return dy * dx;
}


