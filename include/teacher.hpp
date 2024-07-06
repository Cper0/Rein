#pragma once

#include<torch/torch.h>
#include"autoencoder.hpp"

class Teacher
{
public:
	explicit Teacher();

	double eval(torch::Tensor& state);

	void learn(torch::Tensor& history);

private:
	std::shared_ptr<AutoEncoder> model;
	std::shared_ptr<torch::optim::SGD> optimizer;
};
