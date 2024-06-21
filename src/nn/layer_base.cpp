#include"layers.hpp"

LayerBase::LayerBase()
{
}

LayerBase::~LayerBase()
{
}

arma::mat LayerBase::forward(arma::mat x)
{
	return x;
}

arma::mat LayerBase::backward(arma::mat x)
{
	return x;
}

