#pragma once

#include<vector>
#include<iostream>
#include<fstream>
#include<string>
#include<ios>
#include<algorithm>
#include<torch/torch.h>


class ImageHolder
{
public:
	ImageHolder(const std::string& path, int rows, int cols, int channels, int indices, int batch_size);

	void reset();
	void shuffle();

	void add(torch::Tensor img);

	torch::Tensor get_batch(int i, torch::Device device);

private:
	std::string file_path;
	std::vector<int> index_map;
	int indices;
	int batch_size;

	int rows, cols, channels;
};
