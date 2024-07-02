#include"sampling.hpp"

SamplingImpl::SamplingImpl()
{

}

torch::Tensor SamplingImpl::forward(torch::Tensor mu, torch::Tensor log_var)
{
	auto size = mu.sizes();
	auto eps = torch::randn(size);
	return mu + torch::exp(0.5 * log_var) * eps;
}
