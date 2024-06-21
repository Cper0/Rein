#pragma once

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<stdexcept>
#include<cstring>
#include<unordered_map>
#include<stack>
#include"rfb/rfbclient.h"
#include"keymap.hpp"

class Monitor
{
public:
	Monitor(const std::string& host, int port);

	bool recieve();
	void close();

	void control(float x, float y, bool l, bool r);

	void push_key(Key106 k);

	const std::vector<unsigned char>& buf() const noexcept { return buffer; }
	int width() const noexcept { return cl->width; }
	int height() const noexcept { return cl->height; }

private:
	rfbClient* cl;
	float pointer_x, pointer_y;
	bool mouse_l, mouse_r;

	std::vector<unsigned char> buffer;

	std::stack<Key106> keystack;
	Keymap keymap;
};
