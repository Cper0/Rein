#include"agent.hpp"

Agent::Agent(double eps)
{
	std::random_device rd;

	epsilon = eps;
	
	engine = std::default_random_engine(rd());

	table = std::vector<double>(AGENT_ACTIONS);
	count = std::vector<int>(AGENT_ACTIONS);

	std::uniform_int_distribution<> act_gen(0, table.size() - 1);
	table[act_gen(engine)] = 0.01;
}

AgentAction Agent::select()
{
	std::uniform_real_distribution<> prob(0, 1);
	const double p = prob(engine);

	if(p < epsilon)
	{
		std::uniform_int_distribution<> act_gen(0, AGENT_ACTIONS - 1);
		
		return static_cast<AgentAction>(act_gen(engine));
	}

	const size_t max_index = std::max_element(table.begin(), table.end()) - table.begin();
	return static_cast<AgentAction>(max_index);
}

void Agent::update(AgentAction act, double reward)
{
	count[act] += 1;

	table[act] = reward / count[act] + (1.0 - 1.0 / count[act]) * table[act];
}
