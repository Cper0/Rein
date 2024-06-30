#pragma once

#include"layer_base.hpp"
#include<random>

class Dropout : public LayerBase
{
public:
	Dropout(double prob);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat x) override;

private:
	std::default_random_engine engine;
	std::uniform_real_distribution<> dist;

	double prob;
	arma::mat mask;
};
