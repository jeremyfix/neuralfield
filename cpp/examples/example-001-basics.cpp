#include <neuralfield.hpp>

// The input type feeding the neural field
using Input = double;

// A function to fill an input activity layer from our Input type
void fillInput(neuralfield::values_iterator begin, 
		neuralfield::values_iterator end,
		const Input& i) {

	for(; begin != end; ++begin) 
		*begin = i + neuralfield::random::uniform(0., 1.);

}

int main(int argc, char* argv[]) {

  {
    // This is the container that will gather all the layers
    auto network = neuralfield::network();
    
    // The input layer fed with our Input datatype
    auto input = neuralfield::layer::input<double>(10, fillInput, "input");
    network += input;
    //input->fill(2.0);
    
    //std::shared_ptr<neuralfield::layer::InputLayer<double> > l = dynamic_cast<std::shared_ptr<neuralfield::layer::InputLayer<double>>>((*network)["input"]);
  }
    // Add a !reference! of this layer to our container
	//network += input;

	// We define a link layer with a gaussian kernel
	// which adds two parameters : amplitude and variance
	//auto exc = neuralfield::layer::gaussian_link1D();
	//network += exc;

	//auto inh = neuralfield::layer::gaussian_link1D();
	//network += inh;

	//auto h = neuralfield::layer::constant();
	//network += h;

	//auto u = neuralfield::integrator::leaky(exc+inh+input+h);
	//network += u;

	//auto fu = neuralfield::transfer::heaviside(u);
	//// 
	//network += fu;


	// The following connections can only be defined when fu is defined
	// but fu requires u which requires exc and inh.... therefore
	// we cannot define these connections when constructing exc and inh
	//
	//exc.connect(fu);
	//inh.connect(fu);

	// Let us a display of the structure of the network
	// this allows to clearly see which parameters every layer needs
	// and how these are ordered in the "master" parameter vector
	//network.pretty_print();
	
	// We can also generate a PDF showing the network 
	//network.metapost();

	// Let us now play a little bit with the network
	// By first feeding an input	
	

	// We need a parameter vector. We can request a correctly size vector
	//auto parameters = {1.0, ....}
	//network.set_parameters(parameters);
	// but may also have initialized our parameters directly
	// auto parameters = {1.0, ......}
	

	// performing some steps
	//network.step(parameters);

	// and getting the fu activities
	//std::cout << "f(u(x)) = " << network["fu"] << std::endl;

	// il nous faut une façon commode :
	// de coller une entrée, de faire un step, de setter des paramètres
	// ça pourrait être pratique d'ajouter des paires [string, layer]
	// pour facilement réaccéder à un layer par la suite en faisant network["fu"]
	//  peut être limiter ça aux layers de valeur ??
}
