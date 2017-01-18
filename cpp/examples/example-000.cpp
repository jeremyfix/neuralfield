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

	// We can instantiate a parametric functional layer
	// providing the parameters directly
	auto g_exc = neuralfield::link::gaussian(1.5, 2., N);
	auto g_inh =  neuralfield::link::gaussian(1.3, 10., N);
	auto fu = neuralfield::function::function("sigmoid", N);
	auto u = neuralfield::layer::leaky_integrator(0.01, N);
	
	g_exc->connect(fu);
	g_inh->connect(fu);
	fu->connect(u);
	u->connect(input);
	
	auto f_layers = std::initializer_list<std::shared_ptr<neuralfield::layer::Layer> >({g_exc, g_inh, fu});
	auto u_layers = std::initializer_list<std::shared_ptr<neuralfield::layer::BufferedLayer> >({u});

	// At init, we need to propagate the potentials through the functions
	for(auto& l: f_layers)
	  l->propagate_values();

	for(unsigned int i = 0 ; i < 1000 ; ++i) {
	  // Updating all the BufferedLayers
	  for(auto& l: u_layers)
	    l->update();

	  // Swapping them all
	  for(auto& l: u_layers)
	    l->swap();

	  // Propagating through all the functional layers
	  for(auto& l: f_layers)
	    l->propagate_values();
	}
	
	std::cout << "Simulation ended" << std::endl;
	std::cout << "u : " << *u << std::endl;

}
