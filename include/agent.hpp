#pragma once

#include<vector>
#include<random>
#include<algorithm>
#include<deque>

#include"agent_actions.hpp"

class Agent
{
public:
	Agent(double eps);

	AgentAction select();
	void update(AgentAction act, double reward);

private:
	double eval_penalty(AgentAction sel);

	std::default_random_engine engine;

	double epsilon;

	std::deque<AgentAction> index;

	std::vector<double> table;
	std::vector<int> count;
};
