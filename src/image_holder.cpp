#include"image_holder.hpp"

ImageHolder::ImageHolder(const std::string& p, int r, int c, int ch, int i, int b) :
	file_path(p),
	index_map(i),
	indices(0),
	batch_size(b),
	rows(r),
	cols(c),
	channels(ch)
{
	std::iota(index_map.begin(), index_map.end(), 0);
}

void ImageHolder::reset()
{
	std::iota(index_map.begin(), index_map.end(), 0);
	indices = 0;
}

void ImageHolder::shuffle()
{
	std::random_shuffle(index_map.begin(), index_map.end());
}

void ImageHolder::add(torch::Tensor img)
{
	if(indices >= index_map.size()) throw std::runtime_error("out of holder's capacity.");

	if(!img.is_contiguous()) img = img.contiguous();

	std::ofstream stream(file_path, std::ios_base::binary | std::ios_base::app);
	if(!stream) throw std::runtime_error("The file couldn't be opened");

	stream.write(reinterpret_cast<char*>(img.data_ptr<float>()), sizeof(float) * img.numel());

	stream.close();

	indices += 1;
}

torch::Tensor ImageHolder::get_batch(int batch_i, torch::Device device)
{
	if(batch_i >= index_map.size() / batch_size) throw std::runtime_error("out of range that holder can treat batches.");

	const int img_size = channels * rows * cols;
	const int img_bytes = sizeof(float) * channels * rows * cols;

	std::vector<float> buffer(img_size);
	torch::Tensor tensor = torch::zeros({batch_size, channels, rows, cols}, device);

	std::ifstream stream(file_path, std::ios_base::binary);
	if(!stream) throw std::runtime_error("The file couldn't be opened");
	for(int i = 0; i < batch_size; i++)
	{
		const int dest = index_map[batch_i * batch_size + i];
		
		stream.seekg(dest * img_bytes, std::ios_base::beg);
		stream.read(
				reinterpret_cast<char*>(buffer.data()),
				img_bytes
		);

		tensor.index({i}) = torch::from_blob(buffer.data(), {channels, rows, cols}, torch::kFloat);
	}

	stream.close();

	return tensor.clone();
}
