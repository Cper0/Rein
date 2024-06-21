#pragma once

#include"keyboard.hpp"
#include<unordered_map>

class Keymap
{
public:
	Keymap();

	unsigned int operator[](Key106 i) const { return map.at(i); }

private:
	std::unordered_map<Key106, unsigned int> map;
};
