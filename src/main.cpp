#include<rfb/rfbclient.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<random>
#include"mnist.hpp"
#include"monitor.hpp"
#include"layers.hpp"
#include"optimizer.hpp"

constexpr int TRIES = 10000;

std::vector<int> table(TRIES);

Mnist mnist(
	"mnist/train-images-idx3-ubyte",
	"mnist/train-labels-idx1-ubyte",
	"mnist/t10k-images-idx3-ubyte",
	"mnist/t10k-labels-idx1-ubyte"
);

std::vector<LayerBase*> layers = {
	new Affine(28 * 28, 50),
	new Relu(50),

	new Affine(50, 10)
};

SoftmaxWithLoss last(10);

int correct = 0;

arma::vec num2onehot(uint8_t n)
{
	arma::vec out(10, arma::fill::zeros);
	for(size_t i = 0; i < 10; i++) out[i] = (i == n) ? 1 : 0;
	return out;
}

double predict(arma::vec x, arma::vec t)
{
	for(auto&& e : layers)
	{
		x = e->forward(x);
	}
	const auto e = last.forward(x, t);

	return e;
}

void backprop()
{
	arma::vec dx = last.backward(1.0);
	for(auto it = layers.rbegin(); it != layers.rend(); it++)
	{
		dx = (*it)->backward(dx);
	}

	((Affine*)layers[0])->optimize();
	((Affine*)layers[2])->optimize();
}

void test()
{
	int correctness = 0;
	for(auto it = table.begin(); it != table.end(); it++)
	{
		arma::mat img;
		const auto label = mnist.get_test_sample(*it, img);
		
		const arma::vec t = num2onehot(label);
		arma::vec x = arma::vectorise(img);

		for(auto&& e : layers) x = e->forward(x);

		const uint8_t index = arma::index_max(x);
		if(index == label)
		{
			correctness++;
		}
	}

	std::cout << "correctness:" << correctness * 100 / table.size() << "%" << std::endl;
}

arma::mat gradient(arma::vec x, arma::vec t)
{
	arma::vec dx = last.backward(1.0);
	for(auto it = layers.rbegin(); it != layers.rend(); it++)
	{
		dx = (*it)->backward(dx);
	}

	return ((Affine*)layers[2])->getdW();
}

arma::mat numerical_gradient(arma::vec x, arma::vec t)
{
	constexpr double EPS = 0.001;

	arma::mat& W = ((Affine*)layers[2])->getW();

	const int width = W.n_cols;
	const int height = W.n_rows;

	arma::mat out(height, width);

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			const double pre = predict(x, t);
			W(i,j) += EPS;
			const double post = predict(x, t);
			W(i,j) -= EPS;

			out(i,j) = (post - pre) / EPS;	
		}
	}

	return out;
}

void check_gradient()
{
	arma::mat img;
	const auto label = mnist.get_train_sample(0, img);
	
	const arma::vec x = arma::vectorise(img);
	const arma::vec t = num2onehot(label);

	predict(x,t);
	const auto bp_W = gradient(x, t);
	const auto nm_W = numerical_gradient(x, t);

	const auto e = arma::accu(arma::abs(bp_W - nm_W)) / bp_W.n_elem;
	std::cout << "e:" << e << std::endl;

}

int main(int argc, char** argv)
{

	std::random_device dev;
	std::default_random_engine engine(dev());

	arma::arma_rng::set_seed_random();

	for(int i = 0; i < TRIES; i++) table[i] = i;
	std::shuffle(table.begin(), table.end(), engine);

	for(auto it = table.begin(); it != table.end(); it++)
	{
		arma::mat img;
		const auto label = mnist.get_train_sample(*it, img);
		
		const arma::vec x = arma::vectorise(img);
		const arma::vec t = num2onehot(label);

		const double err = predict(x, t);
		backprop();

		//std::cout << err << "," << std::endl;
	}

	test();

	//check_gradient();




	const std::string host = "localhost";
	const int port = 5900;

	std::uniform_real_distribution<> dist(0, 1);

	Monitor monitor(host, port);
	while(monitor.recieve())
	{
		const float mx = std::round(dist(engine)) * 10 - 5;
		const float my = std::round(dist(engine)) * 10 - 5;
		monitor.control(mx, my, dist(engine) >= 0.5, dist(engine) >= 0.5);

	}

	monitor.close();



	std::cout << "Hello World" << std::endl;
	return 0;
}
