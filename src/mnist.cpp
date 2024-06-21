#include"mnist.hpp"
#include<iostream>
#include<fstream>


unsigned int byteswap(unsigned int x)
{
	unsigned int y = 0;
	y |= (x & 0x000000ff) << 24;
	y |= (x & 0x0000ff00) << 8;
	y |= (x & 0x00ff0000) >> 8;
	y |= (x & 0xff000000) >> 24;

	return y;
}


Mnist::Mnist(const std::string& train_img, const std::string& train_lab, const std::string& test_img, const std::string& test_lab)
{
	const auto [a, b, c] = load_images(train_img, train_images);
	load_labels(train_lab, train_labels);
	train_samples = a;
	train_img_rows = b;
	train_img_cols = c;
	
	const auto [d, e, f] = load_images(test_img, test_images);
	load_labels(test_lab, test_labels);
	test_samples = d;
	test_img_rows = e;
	test_img_cols = f;
}

uint8_t Mnist::get_train_sample(size_t i, arma::mat& out) const
{
	const size_t img_size = train_img_rows * train_img_cols;
	
	out = arma::mat(train_img_rows, train_img_cols);

	for(size_t y = 0; y < train_img_rows; y++)
	{
		for(size_t x = 0; x < train_img_cols; x++)
		{
			out(y, x) = train_images[i * img_size + y * train_img_cols + x];
		}
	}

	return train_labels[i];
}

uint8_t Mnist::get_test_sample(size_t i, arma::mat& out) const
{
	const size_t img_size = test_img_rows * test_img_cols;
	
	out = arma::mat(test_img_rows, test_img_cols);

	for(size_t y = 0; y < test_img_rows; y++)
	{
		for(size_t x = 0; x < test_img_cols; x++)
		{
			out(y, x) = test_images[i * img_size + y * test_img_cols + x];
		}
	}

	return test_labels[i];
}

uint8_t Mnist::load_labels(const std::string& p, std::vector<unsigned char>& out)
{
	std::ifstream s(p, std::ios::binary);
	if(!s) throw std::runtime_error("There is no '" + p + "'.");

	unsigned int magic_number, length;

	s.read((char*)&magic_number, sizeof(magic_number));
	s.read((char*)&length, sizeof(length));
	length = byteswap(length);

	out = std::vector<unsigned char>(length);

	s.read((char*)out.data(), sizeof(unsigned char) * length);

	s.close();

	return length;
}

std::tuple<uint32_t,uint32_t,uint32_t> Mnist::load_images(const std::string& p, std::vector<double>& out)
{
	std::ifstream s(p, std::ios::binary);
	if(!s) throw std::runtime_error("There is no '" + p + "'.");

	unsigned int magic_number, length, rows, cols;

	s.read((char*)&magic_number, sizeof(magic_number));
	s.read((char*)&length, sizeof(length));
	s.read((char*)&rows, sizeof(rows));
	s.read((char*)&cols, sizeof(cols));

	length = byteswap(length);
	rows = byteswap(rows);
	cols = byteswap(cols);

	const unsigned int image_size = rows * cols;

	out = std::vector<double>(length * image_size);
	std::vector<unsigned char> buf(image_size);

	for(size_t i = 0; i < length; i++)
	{
		s.read((char*)buf.data(), sizeof(unsigned char) * image_size);
		
		for(size_t j = 0; j < image_size; j++)
		{
			const size_t offset = i * image_size;
			out[offset + j] = static_cast<double>(buf[j]) / 255;
		}
	}

	s.close();

	return std::make_tuple(length, rows, cols);
}
