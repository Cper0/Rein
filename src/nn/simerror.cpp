#include"simerror.hpp"

#include<cmath>

SimError::SimError(Size in_size)
{
	input_size = in_size;
}

double SimError::forward(arma::mat x, arma::mat t)
{
	in_sub = x - t;
	const double d = arma::accu(in_sub % in_sub) / x.n_elem;
	return std::sqrt(d);
}

arma::mat SimError::backward(double dout)
{
	const double divider = arma::accu(in_sub % in_sub) * input_size.rows * input_size.cols;
	const arma::mat img = in_sub / std::sqrt(divider);
		
	return img * dout;
}

