#include"rectifier.hpp"

Rectifier::Rectifier(size_t in_l, Size out_s)
{
	in_length = in_l;
	out_size = out_s;
}

arma::mat Rectifier::forward(arma::mat x)
{
	x.reshape(out_size.rows, out_size.cols);
	return x.t();
}

arma::mat Rectifier::backward(arma::mat dout)
{
	return arma::vectorise(dout);
}
