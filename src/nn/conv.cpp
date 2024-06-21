#include"layers.hpp"

Convolution::Convolution(size_t input_x, size_t input_y, size_t kernel_x, size_t kernel_y)
{
	input_cols = input_x;
	input_rows = input_y;

	kernel = arma::mat(kernel_y, kernel_x, arma::fill::randu);
	kernel_cols = kernel_x;
	kernel_rows = kernel_y;
}

arma::mat Convolution::forward(arma::mat x)
{
	if(x.n_rows != input_rows || x.n_cols != input_cols)
	{
		throw std::logic_error("");
	}

	const size_t walk_width = x.n_cols - kernel_cols + 1;
	const size_t walk_height = x.n_rows - kernel_rows + 1;

	col = im2mat(x);
	col_W = arma::vectorise(kernel, 1);

	arma::mat out = col_W * col + b;
	out.reshape(walk_height, walk_width);
	out = out.t();

	return out;
}

arma::mat Convolution::backward(arma::mat dy)
{
	dy = arma::vectorise(dy, 1);

	//derivate fitler
	const double db = arma::accu(dy);

	arma::mat dW = col * dy;
	dW.reshape(kernel_rows, kernel_cols);
	dW = dW.t();

	arma::mat dcol = dy * col_W.t();
	arma::mat dx = mat2im(dcol);

	return dx;
}

arma::mat Convolution::im2mat(arma::mat img)
{
	const size_t walk_width = img.n_cols - kernel_cols + 1;
	const size_t walk_height = img.n_rows - kernel_rows + 1;

	arma::mat M(kernel.n_elem, walk_width * walk_height);

	for(size_t i = 0; i < walk_height; i++)
	{
		for(size_t j = 0; j < walk_width; j++)
		{
			for(size_t k = i; k < i + kernel_rows; k++)
			{
				for(size_t l = j; l < j + kernel_cols; l++)
				{
					const auto fy = k - i;
					const auto fx = l - j;
					M(fy * kernel_cols + fx, i * walk_width + j) = img(k, l);
				}
			}
		}
	}

	return M;
}

arma::mat Convolution::mat2im(arma::mat col)
{
	/*
	 * col = (out_size, kernel_size)
	*/

	const size_t walk_width = input_cols - kernel_cols + 1;
	const size_t walk_height = input_rows - kernel_rows + 1;

	arma::mat img(input_rows, input_cols);
	for(size_t i = 0; i < input_rows; i++)
	{
		for(size_t j = 0; j < input_cols; j++)
		{
			for(size_t y = 0; y < walk_height; y++)
			{
				for(size_t x = 0; x < walk_width; x++)
				{
					img(y + i, x + j) += col(y * walk_width + x, i * input_cols + j);
				}
			}
		}
	}

	return img;



}

