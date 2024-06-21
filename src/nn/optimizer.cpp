#include"optimizer.hpp"

#include<fstream>
#include<iostream>

Optimizer::Optimizer()
{
	std::ifstream fs("./param");
	if(!fs) throw std::exception();

	fs >> l;

	a = 0.9;
	v = 0;

	fs.close();
}

double Optimizer::optimize(double x)
{
	return l * x;
}

arma::mat Optimizer::optimize(arma::mat X)
{
	return X * l;
}
