#pragma once

#include<vector>
#include"keyboard.hpp"

struct Action
{
	float x, y;
	bool left, right;
	bool keys[KEY106_TOTAL];
};

class Agent
{
public:
	

	void generate_image(std::vector<unsigned char>& buffer);
	void act(const std::vector<unsigned char>& buffer, Action& action);
	
};
