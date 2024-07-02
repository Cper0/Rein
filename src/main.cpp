#include<rfb/rfbclient.h>

#include<iostream>
#include<cassert>
#include<random>
#include<armadillo>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<torch/torch.h>


#include"monitor.hpp"
#include"autoencoder.hpp"
#include"agent.hpp"
#include"teacher.hpp"

torch::Tensor matToTensor(cv::Mat input) {
	cv::Mat mat;
	input.convertTo(mat, CV_64FC3);

    torch::Tensor tensor;
    if (mat.type() == CV_64FC3) {
        tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, 3}, torch::kFloat);
    } else {
        throw std::runtime_error("Unsupported cv::Mat type");
    }

    tensor = tensor.permute({2, 0, 1});

    return tensor.clone(); // 安全のためクローンを返す
}

cv::Mat tensorToMat(torch::Tensor t)
{
	const auto quant = t.mul(255).clamp(0, 255).to(torch::kU8);

	const int rows = t.size(1);
	const int cols = t.size(2);
	const int channels= t.size(0);

	const auto arranged = quant.permute({1, 2, 0});
	const cv::Mat out(cv::Size(rows, cols), CV_8UC3, arranged.data_ptr());
	return out;
}

cv::Mat get_screen_mat(Monitor& monitor)
{
	std::vector<unsigned char> buffer(monitor.width() * monitor.height() * 4);
	std::memcpy(buffer.data(), monitor.frame_buffer(), buffer.size());
	cv::Mat screen(monitor.height(), monitor.width(), CV_8UC4, buffer.data());

	cv::Mat compressed;
	cv::resize(screen, compressed, cv::Size(512, 512), cv::INTER_LINEAR);

	cv::Mat without_alpha;
	cv::cvtColor(compressed, without_alpha, cv::COLOR_BGRA2BGR);

	return without_alpha;
}

int main(int argc, char** argv)
{
	torch::manual_seed(1);


	cv::namedWindow("window");

	constexpr int EXPLORING_TIMES = 1000;

	int times = 0;

	Teacher teacher = Teacher();
	Agent agent(0.1);

	torch::Tensor history = torch::zeros({EXPLORING_TIMES, 3, 512, 512});

	Monitor monitor("localhost", 5900);
	while(monitor.recieve())
	{
		if(times == EXPLORING_TIMES)
		{
			teacher.learn(history);
			times = 0;
			continue;
		}

		const AgentAction act = agent.select();

		switch(act)
		{
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

		cv::Mat img = get_screen_mat(monitor);

		torch::Tensor screen = matToTensor(img);

		history.index({times}) = screen;


		const double reward = teacher.eval(screen);
		agent.update(act, reward);

		std::cout << "reward:" << reward << std::endl;

		times++;

		cv::Mat view;
		cv::cvtColor(img, view, cv::COLOR_BGR2BGRA);
		
		cv::imshow("window", view);
		cv::waitKey(1);
	}

	monitor.close();



	std::cout << "Hello World" << std::endl;
	return 0;
}
