#pragma once

#include<armadillo>
#include"layer_base.hpp"

class SimError
{
public:
	SimError(Size input_size);

	double forward(arma::mat x, arma::mat t);
	arma::mat backward(double dout);

private:
	Size input_size;
	arma::mat in_sub;
};
