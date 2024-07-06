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


int main(int argc, char** argv) {
    torch::manual_seed(1);

    cv::namedWindow("window");

    constexpr int EXPLORING_TIMES = 100;
	constexpr double AGENT_EPSILON = 0.05;

    int times = 0;

    Teacher teacher = Teacher();
    Agent agent(AGENT_EPSILON);

	torch::Tensor history = torch::zeros({EXPLORING_TIMES, 3, 512, 512}, torch::Device(torch::kCUDA));

	cv::cuda::GpuMat origin, squared, true_color, norm;

    Monitor monitor("localhost", 5900);
    while(monitor.recieve()) {
        if(times == EXPLORING_TIMES) {
			teacher.learn(history);
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

		auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
		torch::Tensor tensor = torch::from_blob(norm.data, {norm.rows, norm.cols, 3}, options);
		tensor = tensor.permute({2, 0, 1});

		history.index({times}) = tensor;

        const double reward = teacher.eval(tensor);
        agent.update(act, reward);

        std::cout << "(" << times << ") reward:" << reward << ",act:" << act << std::endl;

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

