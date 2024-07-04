#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <iostream>

#include"monitor.hpp"
#include"autoencoder.hpp"
#include"agent.hpp"
#include"teacher.hpp"

double eval_penalty(const std::deque<AgentAction>& actions, AgentAction sel, double base_penalty)
{
	double weight = 1;
	double result = 0;
	for(auto it = actions.begin(); it != actions.end(); it++)
	{
		if(*it == sel)
		{
			result += base_penalty * weight;
		}

		weight /= 2.0;
	}

	return result;
}



int main(int argc, char** argv) {
    torch::manual_seed(1);

    cv::namedWindow("window");

    constexpr int EXPLORING_TIMES = 20;

    int times = 0;

    Teacher teacher = Teacher();
    Agent agent(0.1);

    torch::Tensor history = torch::zeros({EXPLORING_TIMES, 3, 512, 512});

	std::deque<AgentAction> actions(10, AGENT_ACTIONS);

	cv::cuda::GpuMat origin, squared, true_color, norm;

    Monitor monitor("localhost", 5900);
    while(monitor.recieve()) {
        if(times == EXPLORING_TIMES) {
			for(int i = 0; i < EXPLORING_TIMES; i++) teacher.learn(history.index({i}));
            times = 0;
            continue;
        }

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

		cv::Mat frame(monitor.height(), monitor.width(), CV_8UC4, monitor.frame_buffer());
		origin.upload(frame);

		cv::cuda::resize(origin, squared, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
		cv::cuda::cvtColor(squared, true_color, cv::COLOR_BGRA2BGR);
    	true_color.convertTo(norm, CV_64FC3);

		cv::Mat mat;
		norm.download(mat);
		torch::Tensor tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, 3}, torch::kFloat);
		tensor = tensor.permute({2, 0, 1});

		history.index({times}) = tensor;

        const double reward = teacher.eval(tensor);
		const double penalty = eval_penalty(actions, act, 0.5);
        agent.update(act, reward - penalty);

		actions.erase(actions.end() - 1);
		actions.push_front(act);

        std::cout << "(" << times << ") reward:" << reward << ",penalty:" << penalty << std::endl;

        times++;

		{
			cv::Mat img;
			squared.download(img);
			cv::imshow("window", img);
			cv::waitKey(1);
		}
    }

    monitor.close();

    std::cout << "Hello World" << std::endl;
    return 0;
}

