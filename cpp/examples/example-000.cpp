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

	auto g1 = neuralfield::function::gaussian_link1D();
	g1.connect(input);

	g1.update();
	

	auto g2 = neuralfield::function::gaussian_link1D(1.0, 3., 0.9, 10.);
	g2.connect(input);
	
	/*
	auto g = neuralfield::layer::gaussian_link1D();
	auto output = neuralfield::layer::values(N);
	
	
	g.set_parameters({1.0, 5.0});
	g.connect(input);
	g.evaluate(output);
	
	std::cout << "Output: " << output << std::endl;
	*/
}
