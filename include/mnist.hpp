#pragma once

#include<string>
#include<vector>
#include<tuple>
#include<armadillo>

class Mnist
{
public:
	Mnist(const std::string&, const std::string&, const std::string&, const std::string&);

	uint8_t get_train_sample(size_t i, arma::mat& out) const;
	uint8_t get_test_sample(size_t i, arma::mat& out) const;
	

private:
	uint8_t load_labels(const std::string&, std::vector<unsigned char>&);
	std::tuple<uint32_t, uint32_t,uint32_t> load_images(const std::string&, std::vector<double>&);

	std::vector<unsigned char> train_labels;
	std::vector<double> train_images;
	uint32_t train_samples;
	uint32_t train_img_rows;
	uint32_t train_img_cols;

	std::vector<unsigned char> test_labels;
	std::vector<double> test_images;
	uint32_t test_samples;
	uint32_t test_img_rows;
	uint32_t test_img_cols;
};
