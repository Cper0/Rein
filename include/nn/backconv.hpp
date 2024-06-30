#pragma once

#include"layer_base.hpp"
#include"conv.hpp"

class BackConvolution : public LayerBase
{
public:
	BackConvolution(Size img_size, Size kernel_size, size_t stride, size_t padding);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat x) override;

	void optimize() override;

private:
	Convolution internal_conv;

	Size input_size;

	size_t stride;
	size_t padding;
};
