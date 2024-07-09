#pragma once

#include<iostream>
#include<vector>
#include<stdexcept>
#include<cstring>
#include<unordered_map>
#include<stack>

#include<torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"

#include"rfb/rfbclient.h"
#include"keymap.hpp"

class Monitor
{
public:
	Monitor(const std::string& host, int port, const std::string& pass = "headless");

	torch::Tensor frame_to_tensor();

	bool recieve();
	void close();

	void control(float x, float y, bool l, bool r);

	void push_key(Key106 k);

	int width() const noexcept { return cl->width; }
	int height() const noexcept { return cl->height; }
	const std::string& password() const noexcept { return pwd; }

private:
	static char* get_password_callback(rfbClient* cl);

	rfbClient* cl;
	float pointer_x, pointer_y;
	bool mouse_l, mouse_r;

	std::string pwd;

	std::stack<Key106> keystack;
	Keymap keymap;
};
