#pragma once

#include<cmath>
#include<armadillo>

namespace util
{
	double sigmoid(double x);
	arma::mat sigmoid(arma::mat x);

	arma::vec softmax(arma::vec x);

	double cross_entropy_error(arma::vec x, arma::vec t);
}
