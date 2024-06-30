#pragma once

#include"layer_base.hpp"
#include"optimizer.hpp"


class Convolution : public LayerBase
{
public:
	Convolution(Size img_size, Size kernel_size, size_t stride, size_t padding);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;
	
	void optimize() override;

private:
	arma::mat im2mat(arma::mat img);
	arma::mat mat2im(arma::mat mat);

	size_t img_rows;
	size_t img_cols;

	size_t stride;
	size_t padding;

	Optimizer W_opt;
	Optimizer b_opt;

			
	arma::mat in_col;

	arma::mat W;
	double b;

	arma::mat dW;
	double db;

};
