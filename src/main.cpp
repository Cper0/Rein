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

    constexpr int EXPLORING_TIMES = 100;
	constexpr double AGENT_EPSILON = 0.05;

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
                monitor.control(-10, 0, false, false);
                break;
            case AGENT_MOUSE_RIGHT:
                monitor.control(10, 0, false, false);
                break;
            case AGENT_MOUSE_UP:
                monitor.control(0, -10, false, false);
                break;
            case AGENT_MOUSE_DOWN:
                monitor.control(0, 10, false, false);
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
			{
				vae->train();
				vae_optim.zero_grad();

				auto trg = screen_history;
				auto out = vae->forward(trg);
				auto [mu, log_var] = vae->encoder->forward(trg);
				auto z = VAEImpl::reparameterize(mu, log_var);

				auto kld = -0.5 * torch::sum(1.0 + log_var - mu.pow(2) - log_var.exp(), 1);
				auto mse = torch::mean(torch::pow(out - trg, 2), {1, 2, 3});
				auto loss = 0.04 * kld + mse;

				loss.sum().backward();

				vae_optim.step();
			}

			{
				vae->eval();
				action_encoder->train();
				act_optim.zero_grad();

				auto target = action_history;
				auto output = vae->decoder->forward(action_encoder->forward(target));
				
				auto loss = torch::mse_loss(output, screen_history);

				loss.sum().backward();
				act_optim.step();
				
			}
			times = 0;
			
			continue;
		}

		auto screen = monitor.frame_to_tensor();
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
			/*
			cv::Mat img;
			squared.download(img);
			cv::imshow("window", img);
			cv::waitKey(1);
			*/
		}
    }

    monitor.close();

    std::cout << "Hello World" << std::endl;
    return 0;
}

