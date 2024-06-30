#include<rfb/rfbclient.h>

#include<iostream>
#include<random>
#include<armadillo>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

#include"monitor.hpp"

#include"mnist.hpp"
#include"layers.hpp"
#include"conv.hpp"
#include"backconv.hpp"
#include"rectifier.hpp"
#include"normalizer.hpp"
#include"simerror.hpp"
#include"dropout.hpp"

std::vector<LayerBase*> encoder = {
	new Convolution({28, 28}, {5, 5}, 2, 2),
	new Relu(0),

	new Convolution({14, 14}, {5, 5}, 2, 2),
	new Relu(0),

	new Flatten(7, 7),

	new Affine(7 * 7, 24),
	new Relu(0),

	new Affine(24, 12),
	new Relu(0),

	new Affine(12, 2),
};

std::vector<LayerBase*> decoder = {
	new Affine(2, 12),
	new Relu(0),

	new Affine(12, 24),
	new Relu(0),

	new Affine(24, 49),
	new Relu(0),

	new Rectifier(7 * 7, {7, 7}),

	new BackConvolution({7, 7}, {5, 5}, 2, 11),
	new Relu(0),

	new BackConvolution({14, 14}, {5, 5}, 2, 21),
	new Sigmoid(0),
};

SimError simerror({28, 28});

std::pair<double,double> encode(arma::mat img)
{
	arma::mat x = img;
	for(size_t i = 0; i < encoder.size(); i++)
	{
		x = encoder[i]->forward(x);
	}

	return std::make_pair(x[0], x[1]);
}

arma::mat decode(std::pair<double,double> pos)
{
	arma::mat x = arma::vec{pos.first, pos.second};
	for(size_t i = 0; i < decoder.size(); i++)
	{
		x = decoder[i]->forward(x);
	}

	return x;
}


int main(int argc, char** argv)
{
	std::random_device dev;
	std::default_random_engine engine(dev());

	Mnist mnist(
		"mnist/train-images-idx3-ubyte",
		"mnist/train-labels-idx1-ubyte",
		"mnist/t10k-images-idx3-ubyte",
		"mnist/t10k-labels-idx1-ubyte"
	);

	for(int i = 0; i < 60000; i++)
	{
		arma::mat img;
		mnist.get_train_sample(0, img);
		
		const auto pos = encode(img);
		const auto out = decode(pos);
		const double err = simerror.forward(out, img);

		std::cout << "[" << i << "] (" << pos.first << "," << pos.second << "),error=" << err << std::endl;

		arma::mat dout = simerror.backward(1);
		for(auto it = decoder.rbegin(); it != decoder.rend(); it++)
		{
			dout = (*it)->backward(dout);
			//std::cout << dout << std::endl;
		}
		for(auto it = encoder.rbegin(); it != encoder.rend(); it++)
		{
			dout = (*it)->backward(dout);
			//std::cout << dout << std::endl;
		}

		for(auto&& e : encoder) e->optimize();
		for(auto&& e : decoder) e->optimize();

		if(out.has_nan())
		{
			std::cout << i << std::endl;
			break;
		}

		//std::cout << out << std::endl;
	}

	/*
	std::uniform_real_distribution<> dist(-10, 10);
	for(;;)
	{
		std::pair<double,double> pos = {dist(engine), dist(engine)};
		const auto out = decode(pos);

		std::vector<double> buffer = arma::conv_to<std::vector<double>>::from(arma::vectorise(out));
		
		cv::Mat viewer(28, 28, CV_64FC1, &buffer);
		cv::Mat dst;
		cv::resize(viewer, dst, cv::Size(), 10, 10, cv::INTER_NEAREST);
		std::cout << out << std::endl;
		
		cv::imshow("window", dst);
		cv::waitKey(0);
	}



	const std::string host = "localhost";
	const int port = 5900;

	Monitor monitor(host, port);
	while(monitor.recieve())
	{
		const float mx = std::round(dist(engine)) * 10 - 5;
		const float my = std::round(dist(engine)) * 10 - 5;
		monitor.control(mx, my, dist(engine) >= 0.5, dist(engine) >= 0.5);

	}

	monitor.close();
	*/



	std::cout << "Hello World" << std::endl;
	return 0;
}
