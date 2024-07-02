#include"normalizer.hpp"

Normalizer::Normalizer()
{
}

arma::mat Normalizer::forward(arma::mat x)
{
	const double avg = arma::mean(arma::mean(x));
	variance = arma::var(arma::var(x));

	return (x - avg) / (variance + 0.01);
}

arma::mat Normalizer::backward(arma::mat dout)
{
	return dout / (variance + 0.01);
}
