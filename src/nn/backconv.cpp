#include"backconv.hpp"

BackConvolution::BackConvolution(Size img_size, Size kernel_size, size_t st, size_t pad) : LayerBase(), internal_conv(img_size + pad * 2, kernel_size, st, kernel_size.rows / 2)
{
	input_size = img_size;
	stride = st;
	padding = pad;
}

arma::mat BackConvolution::forward(arma::mat input)
{
	const auto new_size = arma::size(input) + 2 * padding;

	const size_t step_x = new_size[0] / input.n_cols;
	const size_t step_y = new_size[1] / input.n_rows;


	arma::mat m(new_size);
	for(size_t y = 0; y < input.n_rows; y++)
	{
		for(size_t x = 0; x < input.n_cols; x++)
		{
			m(y * step_x, x * step_y) = input(y,x);
		}
	}

	return internal_conv.forward(m);
}

arma::mat BackConvolution::backward(arma::mat dout)
{
	arma::mat d = internal_conv.backward(dout);

	const auto new_size = input_size + padding * 2;
	const size_t step_x = new_size.cols / input_size.cols;
	const size_t step_y = new_size.rows / input_size.rows;

	arma::mat m(input_size.rows, input_size.cols);
	for(size_t y = 0; y < input_size.rows; y++)
	{
		for(size_t x = 0; x < input_size.cols; x++)
		{
			m(y,x) = d(y * step_x, x * step_y);
		}
	}

	return m;
}

void BackConvolution::optimize()
{
	internal_conv.optimize();
}
