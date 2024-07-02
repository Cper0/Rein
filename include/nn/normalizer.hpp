#pragma once

#include"layer_base.hpp"

class Normalizer : public LayerBase
{
public:
	Normalizer();

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat x) override;
private:
	double variance;
};
