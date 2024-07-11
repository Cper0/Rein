#include <iostream>
#include<algorithm>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"

#include"monitor.hpp"
#include"agent.hpp"
#include"action_encoder.hpp"
#include"vae.hpp"
#include"image_holder.hpp"

using namespace torch::indexing;

constexpr int MOVE = 1;
constexpr int EXPLORING_TIMES = 100;
constexpr int BATCH_SIZE = 10;
constexpr int EPOCHS = EXPLORING_TIMES / BATCH_SIZE;
constexpr double AGENT_EPSILON = 0.05;

torch::Device device(torch::kCUDA);

torch::Tensor onehot_action(AgentAction act)
{
	auto x = torch::zeros(static_cast<int>(AGENT_ACTIONS), device);
	x.index({static_cast<int>(act)}) = 1.0;
	return x;
}

int main(int argc, char** argv) {
    torch::manual_seed(1);

    cv::namedWindow("window");

	double lr = 0.001;

	if(argc == 2)
	{
		lr = std::atof(argv[1]);
	}

	cv::namedWindow("window", cv::WINDOW_AUTOSIZE);


    int times = 0;

	auto action_encoder = std::make_shared<ActionEncoder>();
	auto vae = VAE();

	action_encoder->to(device);
	vae->to(device);

	auto act_optim = torch::optim::SGD(action_encoder->parameters(), lr);
	auto vae_optim = torch::optim::SGD(vae->parameters(), lr);

    Agent agent(AGENT_EPSILON);

	torch::Tensor screen_history = torch::zeros({EXPLORING_TIMES, 3, 128, 128}, device);
	torch::Tensor action_history = torch::zeros({EXPLORING_TIMES, AGENT_ACTIONS}, device);

	bool connection = true;

    Monitor monitor("localhost", 5900);
    while(connection)
	{
        const AgentAction act = agent.select();
        switch(act) {
            case AGENT_MOUSE_LEFT:
                monitor.control(-MOVE, 0, false, false);
                break;
            case AGENT_MOUSE_RIGHT:
                monitor.control(MOVE, 0, false, false);
                break;
            case AGENT_MOUSE_UP:
                monitor.control(0, -MOVE, false, false);
                break;
            case AGENT_MOUSE_DOWN:
                monitor.control(0, MOVE, false, false);
                break;
            case AGENT_BUTTON_LEFT:
                monitor.control(0, 0, true, false);
                break;
            case AGENT_BUTTON_RIGHT:
                monitor.control(0, 0, false, true);
                break;
            default:
                break;
        }
		
		connection = monitor.recieve();

		if(times == EXPLORING_TIMES)
		{
			vae->train();
			vae_optim.zero_grad();

			for(int i = 0; i < EPOCHS; i++)
			{
				auto target = screen_history.index({Slice(i * BATCH_SIZE, i * BATCH_SIZE + BATCH_SIZE)});
				auto output = vae->forward(target);
				auto [mu, log_var] = vae->encoder->forward(target);
				auto z = VAEImpl::reparameterize(mu, log_var);

				auto kld = -0.5 * torch::sum(1.0 + log_var - mu.pow(2) - log_var.exp(), 1);
				auto mse = torch::mean(torch::pow(output - target, 2), {1, 2, 3});

				auto loss = 0.04 * kld + mse;

				loss.sum().backward();

				vae_optim.step();
			}

			vae->eval();
			action_encoder->train();
			act_optim.zero_grad();

			for(int i = 0; i < EPOCHS; i++)
			{
				auto actions = action_history.index({Slice(i * BATCH_SIZE, (i+1) * BATCH_SIZE)});
				auto target = screen_history.index({Slice(i*BATCH_SIZE,(i+1)*BATCH_SIZE)});

				auto output = vae->decoder->forward(action_encoder->forward(actions));
				
				auto loss = torch::mse_loss(output, target);

				loss.sum().backward();
				act_optim.step();
			}

			times = 0;
			
			continue;
		}

		auto [screen, img] = monitor.create_image();
		screen_history.index({times}) = screen;
		action_history.index({times}) = onehot_action(act);

		action_encoder->eval();
		vae->eval();

		auto x = onehot_action(act);
		x = action_encoder->forward(x);
		x = vae->decoder->forward(x);

		auto loss = torch::mse_loss(x, screen);

        const double reward = std::exp(-loss.item<double>());
        agent.update(act, reward);

        std::cout << "(" << times << ") loss:" << loss.item<double>() << ",reward:"  << reward << ",act:" << act << std::endl;

        times++;

		{
			cv::imshow("window", img);
			cv::waitKey(1);
		}
    }

    monitor.close();

    std::cout << "Hello World" << std::endl;
    return 0;
}

