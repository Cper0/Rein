#include"teacher.hpp"

Teacher::Teacher()
{
	model = std::make_shared<AutoEncoder>();
	model->to(torch::kCUDA);

	optimizer = std::make_shared<torch::optim::SGD>(model->parameters(), 0.01);
}

double Teacher::eval(torch::Tensor& state)
{
	model->eval();

	state = state.view({-1, 3, 512, 512});

	auto prediction = model->forward(state);
	auto loss = torch::mse_loss(prediction, state);

	return std::exp(-loss.item<double>());
}

void Teacher::learn(torch::Tensor& history)
{
	model->train();

	auto prediction = model->forward(history);
	auto loss = torch::mse_loss(prediction, history);

	loss.backward();
	optimizer->step();


}
