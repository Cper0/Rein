#include"dropout.hpp"

Dropout::Dropout(double p) : LayerBase(), dist(0,1)
{
	prob = p;

	std::random_device rd;
	engine = std::default_random_engine(rd());
}

arma::mat Dropout::forward(arma::mat x)
{
	mask.copy_size(x);
	for(int i = 0; i < x.n_rows; i++)
	{
		for(int j = 0; j < x.n_cols; j++)
		{
			mask(i,j) = (dist(engine) < prob) ? 0 : 1;
		}
	}

	return x % mask;
}

arma::mat Dropout::backward(arma::mat x)
{
	return x % mask;
}

