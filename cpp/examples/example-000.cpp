#include <neuralfield.hpp>

#define N 10

using Input = double;

void fillInput(neuralfield::values_iterator begin,
		neuralfield::values_iterator end,
		const Input& x) {
	int i = 0;
	for(; begin != end; ++begin, ++i) {
		double d2 = (i - x)*(i-x);
		*begin = exp(-d2 / (2.0 * (N/4.0)*(N/4.0)));
	}
}

int main(int argc, char* argv[]) {
	neuralfield::values_type activities;

	auto input = neuralfield::layer::input<Input>(N, fillInput);
	input->fill(N/2);
	
	std::cout << "Input : " << *input << std::endl;

	// A Network is a container of all the layers
	// which will rule the evaluation of the layers
	auto net = neuralfield::network();
	
	// We can instantiate a parametric functional layer
	// providing the parameters directly
	auto g_exc = neuralfield::link::gaussian(1.5, 2., N);
	net += g_exc;
	
	auto g_inh =  neuralfield::link::gaussian(1.3, 10., N);
	net += g_inh;
	
	auto fu = neuralfield::function::function("sigmoid", N, "fu");
	net += fu;
	
	auto u = neuralfield::layer::leaky_integrator(0.01, N);
	net += u;

	// We connect all the layers together
	g_exc->connect(fu);
	g_inh->connect(fu);
	fu->connect(u);
	u->connect(input);


	net->init();
	
	for(unsigned int i = 0 ; i < 10 ; ++i) {
	  net->step();
	}
	
	std::cout << "Simulation ended" << std::endl;
	std::cout << "u : " << *u << std::endl;

}
