#include"layers.hpp"

SoftmaxWithLoss::SoftmaxWithLoss(size_t input_s)
{
	input_size = input_s;
	y = arma::vec(input_size);
	t = arma::vec(input_size);
}

double SoftmaxWithLoss::forward(arma::vec x, arma::vec p)
{
	t = p;
	y = util::softmax(x);

	const auto e = util::cross_entropy_error(y, t);

	return e;
}

arma::vec SoftmaxWithLoss::backward(double dy)
{
	auto e = (y - t) * dy;
	return e;
}
