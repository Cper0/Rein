#include"action_encoder.hpp"

ActionEncoder::ActionEncoder()
{
	dense[0] = register_module("dense0", torch::nn::Linear(AGENT_ACTIONS, 25));
	norm[0] = register_module("norm0", torch::nn::LayerNorm(torch::nn::LayerNormOptions({25})));

	dense[1] = register_module("dense1", torch::nn::Linear(25, 50));
	norm[1] = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({50})));

	dense[2] = register_module("dense2", torch::nn::Linear(50, 100));
	norm[2] = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({100})));

	dense[3] = register_module("dense3", torch::nn::Linear(100, 200));
	//norm[3] = register_module("norm3", torch::nn::BatchNorm1d(200));
}

torch::Tensor ActionEncoder::forward(torch::Tensor x)
{
	x = x.view({-1, AGENT_ACTIONS});

	for(int i = 0; i < 3; i++)
	{
		x = dense[i]->forward(x);
		x = norm[i]->forward(x);
		x = torch::leaky_relu(x);
	}

	x = dense[3]->forward(x);
	x = torch::sigmoid(x);

	return x;
}
