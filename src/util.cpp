#include"util.hpp"

#include<cmath>
#include<armadillo>

namespace util
{
	double sigmoid(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	arma::vec sigmoid(arma::vec x)
	{
		return 1.0 / (1.0 + arma::exp(-x));
	}

	arma::vec softmax(arma::vec x)
	{
		x = arma::exp(x - arma::max(x));
		const double s = arma::accu(x);

		return x / s;
	}

	double cross_entropy_error(arma::vec x, arma::vec t)
	{
		const size_t max_index = arma::index_max(t);
		const double y = -std::log(x[max_index]);
		return y;
	}
}

