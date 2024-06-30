#include"layer_base.hpp"
#include<iostream>

LayerBase::LayerBase()
{
}

LayerBase::~LayerBase()
{
}

arma::mat LayerBase::forward(arma::mat x)
{
	std::cout << "oh... hello? I'm from 'forward'" << std::endl;
	return x;
}

arma::mat LayerBase::backward(arma::mat x)
{
	std::cout << "oh... hello? I'm from 'backward'" << std::endl;
	return x;
}


void LayerBase::optimize()
{
	std::cout << "oh... hello?" << std::endl;
}
