#include"layers.hpp"
#include"optimizer.hpp"

Affine::Affine(size_t input_s, size_t output_s) : LayerBase(), opt_W(), opt_b()
{
	input_size = input_s;
	output_size = output_s;

	W = arma::mat(output_size, input_size, arma::fill::randu) * 0.01;
	b = 0; // should be random value
}

Affine::Affine(size_t input_s, size_t output_s, Optimizer* o) : LayerBase(), opt_W(), opt_b()
{
	input_size = input_s;
	output_size = output_s;

	W = arma::mat(output_size, input_size, arma::fill::randu);
	b = 0; // should be random value;
	
	opt = o;
}

arma::mat Affine::forward(arma::mat x)
{
	if(x.n_rows != input_size || x.n_cols != 1)
	{
		std::stringstream st;
		st << "Error was thrown on forwarding in Affine layer.\n";
		st << "X is (" << x.n_elem << ")";
		throw std::logic_error(st.str());
	}

	in = x;
	return W * x + b;
}

arma::mat Affine::backward(arma::mat dy)
{
	const arma::vec dx = W.t() * dy;
	dW = dy * in.t();
	db = arma::accu(dy) / dy.n_elem;

	/*
	W += opt_W.optimize(dW);
	b += opt_b.optimize(db);
	*/

	return dx;
}

void Affine::optimize()
{
	W += opt_W.optimize(dW);
	b += opt_b.optimize(db);
}
