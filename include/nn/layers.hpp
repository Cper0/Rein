#pragma once

#include<armadillo>
#include<sstream>
#include"util.hpp"
#include"conv.hpp"
#include"backconv.hpp"
#include"optimizer.hpp"


class Relu : public LayerBase
{
public:
	Relu(size_t);
	
	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;

private:
	size_t input_size;
	arma::mat mask;
};

class Sigmoid : public LayerBase
{
public:
	Sigmoid(size_t);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;

private:
	size_t input_size;
	arma::mat out;
};

class Affine : public LayerBase
{
public:
	Affine(size_t, size_t);
	Affine(size_t, size_t, Optimizer*);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;
	void optimize() override;

	arma::mat& getW() { return W; }
	const arma::mat& getdW() { return dW; }

private:
	size_t input_size;
	size_t output_size;

	arma::vec in;

	arma::mat W;
	double b;

	arma::mat dW;
	double db;

	Optimizer* opt;
	Optimizer opt_W;
	Optimizer opt_b;
};

class Flatten : public LayerBase
{
public:
	Flatten(size_t in_rows, size_t in_cols);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat x) override;

private:
	size_t input_rows;
	size_t input_cols;
};

class SME
{
public:
	SME(size_t);

	double forward(arma::vec x, arma::vec t);
	arma::vec backward(double dy);

private:
	size_t input_size;
	arma::vec in_x;
	arma::vec in_t;
};

class SoftmaxWithLoss
{
public:
	SoftmaxWithLoss(size_t);
	
	double forward(arma::vec x, arma::vec p);
	arma::vec backward(double dy);

private:
	size_t input_size;
	arma::vec t;
	arma::vec y;
};

