#include"teacher.hpp"

Teacher::Teacher() : model(std::make_shared<AutoEncoder>()), optimizer(model->parameters(), 0.01)
{
}

double Teacher::eval(torch::Tensor state)
{
	model->eval();

	auto prediction = model->forward(state);
	auto loss = torch::mse_loss(prediction, state);

	return std::exp(-loss.item<double>());
}

void Teacher::learn(torch::Tensor history)
{
	model->train();

	auto prediction = model->forward(history);
	auto loss = torch::mse_loss(prediction, history);

	loss.backward();
	optimizer.step();


}
