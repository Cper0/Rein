#pragma once

#include<armadillo>

struct Size
{
	size_t rows;
	size_t cols;

	Size operator*(size_t x) const noexcept
	{
		return Size{rows * x, cols * x};
	}

	Size operator+(size_t x) const noexcept
	{
		return Size{rows + x, cols + x};
	}

	Size operator/(size_t x) const noexcept
	{
		return Size{rows / x, cols / x};
	}
};

class LayerBase
{
public:
	LayerBase();
	virtual ~LayerBase();

	virtual arma::mat forward(arma::mat x);
	virtual arma::mat backward(arma::mat x);

	virtual void optimize();
};
