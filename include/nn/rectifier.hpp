#pragma once

#include"layer_base.hpp"

class Rectifier : public LayerBase
{
public:
	Rectifier(size_t input_length, Size out_size);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat x) override;
private:
	size_t in_length;
	Size out_size;
};
