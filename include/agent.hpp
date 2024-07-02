#pragma once

#include<vector>
#include<random>
#include<algorithm>

enum AgentAction
{
	AGENT_MOUSE_LEFT,
	AGENT_MOUSE_RIGHT,
	AGENT_MOUSE_UP,
	AGENT_MOUSE_DOWN,
	AGENT_BUTTON_LEFT,
	AGENT_BUTTON_RIGHT,
	AGENT_ACTIONS
};

class Agent
{
public:
	Agent(double eps);

	AgentAction select();
	void update(AgentAction act, double reward);

private:
	std::default_random_engine engine;

	double epsilon;

	std::vector<double> table;
	std::vector<int> count;
};
