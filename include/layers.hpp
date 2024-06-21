#pragma once

#include<armadillo>
#include<sstream>
#include"util.hpp"
#include"optimizer.hpp"

class LayerBase
{
public:
	LayerBase();
	virtual ~LayerBase();

	virtual arma::mat forward(arma::mat x);
	virtual arma::mat backward(arma::mat x);
};

class Relu : public LayerBase
{
public:
	Relu(size_t);
	
	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;

private:
	size_t input_size;
	arma::vec mask;
};

class Sigmoid : public LayerBase
{
public:
	Sigmoid(size_t);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;

private:
	size_t input_size;
	arma::vec out;
};

class Affine : public LayerBase
{
public:
	Affine(size_t, size_t);
	Affine(size_t, size_t, Optimizer*);

	arma::mat forward(arma::mat x) override;
	arma::mat backward(arma::mat dy) override;
	void optimize();

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

class Convolution
{
public:
	Convolution(size_t, size_t, size_t, size_t);

	arma::mat forward(arma::mat x);

	arma::mat backward(arma::mat dy);

private:
	arma::mat im2mat(arma::mat img);
	arma::mat mat2im(arma::mat mat);
			
	arma::mat kernel;
	size_t kernel_rows;
	size_t kernel_cols;
	double b;

	size_t input_rows;
	size_t input_cols;

	arma::mat col;
	arma::rowvec col_W;
};
