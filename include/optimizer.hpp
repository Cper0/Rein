#pragma once


#include<armadillo>

class Optimizer
{
public:
	Optimizer();

	double optimize(double x);
	arma::mat optimize(arma::mat x);

private:
	double l, a, v;
};
