#include"conv.hpp"
#include"optimizer.hpp"

Convolution::Convolution(Size img_size, Size kernel_size, size_t st, size_t pad) : LayerBase()
{
	img_cols = img_size.cols;
	img_rows = img_size.rows;

	W = arma::mat(kernel_size.rows, kernel_size.cols, arma::fill::randu) * 0.01;
	b = 0;

	W_opt = Optimizer();
	b_opt = Optimizer();

	stride = st;
	padding = pad;
}

arma::mat Convolution::forward(arma::mat x)
{
	if(x.n_rows != img_rows || x.n_cols != img_cols)
	{
		throw std::logic_error("");
	}

	arma::mat img(arma::size(x) + 2 * padding, arma::fill::zeros);
	img.submat(padding, padding, padding + x.n_rows - 1, padding + x.n_cols - 1) = x;

	arma::vec kernel = arma::vectorise(W);
	in_col = im2mat(img);

	const auto out_size = (arma::size(x)  + 2 * padding - arma::size(W) + 1) / stride;

	arma::mat out = in_col * kernel + b;
	out.reshape(out_size);
	out = out.t();

	std::cout << out.has_inf() << std::endl;

	return out;
}

arma::mat Convolution::backward(arma::mat dy)
{
	arma::vec dout = arma::vectorise(dy);
	arma::vec kernel = arma::vectorise(W);

	dW = in_col.t() * dout;
	dW.reshape(arma::size(W));
	dW = dW.t();

	db = arma::accu(dy);

	arma::mat dcol = dout * kernel.t();
	arma::mat dimg = mat2im(dcol);

	return dimg;
}

void Convolution::optimize()
{
	W += W_opt.optimize(dW);
	b += b_opt.optimize(db);
}

arma::mat Convolution::im2mat(arma::mat img)
{
	const auto s = (arma::size(img) - arma::size(W) + 1) / stride;

	arma::mat M(s[0] * s[1], W.n_elem);

	for(size_t i = 0; i < s.n_rows; i++)
	{
		for(size_t j = 0; j < s.n_cols; j++)
		{
			const size_t y = i * stride;
			const size_t x = j * stride;

			arma::mat sub = img.submat(y, x, y + W.n_rows - 1, x + W.n_cols - 1);
			arma::vec block = arma::vectorise(sub);
			
			M.row(i * s[1] + j) = block.t();
		}
	}

	return M;
}

arma::mat Convolution::mat2im(arma::mat col)
{
	/*
	 * col = (out_size, kernel_size)
	*/
	const auto s = (arma::size(img_rows, img_cols) + 2 * padding - arma::size(W) + 1) / stride;

	arma::mat img(img_rows + 2 * padding, img_cols + 2 * padding);
	for(size_t i = 0; i < s.n_rows; i++)
	{
		for(size_t j = 0; j < s.n_cols; j++)
		{
			const size_t y = i * stride;
			const size_t x = j * stride;

			arma::mat filter = col.row(i * s.n_cols + j);
			filter.reshape(arma::size(W));
			filter = filter.t();

			img.submat(y, x, y + W.n_rows - 1, x + W.n_cols - 1) += filter;
		}
	}

	return img.submat(padding, padding, padding + img_rows - 1, padding + img_cols - 1);
}

