#include "neuralfield.hpp"
#include <iostream>

using Input = std::vector<double>;

void fillInput(neuralfield::values_iterator begin,
	       neuralfield::values_iterator end,
	       const Input& x) {
  std::copy(x.begin(), x.end(), begin);
}

Input generate_input(std::vector<int> shape,
		     std::string label) {
  int size = 1;
  for(auto s: shape)
    size *= s;

  Input input(size);
  
  if(label == "random") {
    for(auto& v: input)
      v = neuralfield::random::uniform(0.0, 1.0);
  }
  else if(label == "structured") {
    // TODO
  }
  return input;
}

int main(int argc, char * argv[]) {

  if(argc != 2) {
    std::cerr << "Script to optimize a 2D neural field for a competition scenario" << std::endl;
    std::cerr << "Usage : " << argv[0] << " N" << std::endl;
    std::exit(-1);
  }

  int N = atoi(argv[1]);

  double Ap = 1.5;
  double sp = 2.;
  double Am = -1.3;
  double sm = 10.;
  double dt_tau = 0.01;
  bool toric = false;
  unsigned int Nsteps = 100;
  
  // 1D
  std::initializer_list<int> shape({N});
  
  // 2D
  // std::initializer_list<int> shape({N, N});
  
  auto input = neuralfield::input::input<Input>(shape, fillInput, "input");
  auto u = neuralfield::buffered::leaky_integrator(dt_tau, shape, "u");
  auto g_exc = neuralfield::link::gaussian(Ap, sp, toric, shape,"gexc");
  auto g_inh =  neuralfield::link::gaussian(Am, sm, toric, shape, "ginh");
  auto fu = neuralfield::function::function("sigmoid", shape, "fu");

  g_exc->connect(fu);
  g_inh->connect(fu);
  fu->connect(u);
  u->connect(g_exc + g_inh + input);
  
  auto net = neuralfield::get_current_network();
  net->print();

  net->init();

  //// This is how one scenario should goes on
  net->reset();

  auto I = generate_input(shape, "random");
  net->set_input<Input>("input", I);
  
  for(unsigned int i = 0 ; i < Nsteps; ++i)
    net->step();
  ////
  
  return 0;
}
