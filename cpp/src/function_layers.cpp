#include "function_layers.hpp"
#include "network.hpp"

neuralfield::function::Layer::Layer(std::string label,
				    typename parameters_type::size_type number_of_parameters,
				    std::vector<int> shape):
  neuralfield::layer::Layer(label, number_of_parameters, shape) {
}

neuralfield::function::Layer::Layer(const neuralfield::function::Layer& other):
  neuralfield::layer::Layer(other),
  _prevs(other._prevs) {  
}
      
void neuralfield::function::Layer::connect(std::shared_ptr<neuralfield::layer::Layer> prev) {
  _prevs.push_back(prev);
}

bool neuralfield::function::Layer::is_connected() {
  bool connected = true;
  auto it = _prevs.begin();
  while(connected && it != _prevs.end()) {
    connected &= bool(*it);
    ++it;
  }
  return connected;
}
      
void neuralfield::function::Layer::update(void) {
  if(! is_connected()) {
    throw std::runtime_error("The layer named '" + label() + "' has some undefined previous layers.");
  }	
}

bool neuralfield::function::Layer::can_be_evaluated_from(const std::map<std::shared_ptr<neuralfield::layer::Layer>, bool>& evaluation_status) {
  bool can_be_evaluated = true;
  auto it = _prevs.begin();
  while(can_be_evaluated && it != _prevs.end()) {
    can_be_evaluated &= evaluation_status.at(*it);
    ++it;
  }
  return can_be_evaluated;
}
     

neuralfield::function::VectorizedFunction::VectorizedFunction(std::string label,
							      std::function<double(double)> f,
							      std::vector<int> shape):
  neuralfield::function::Layer(label, 0, shape), _f(f) {
}

void neuralfield::function::VectorizedFunction::update() {
  if(_prevs.size() != 1) {
    throw std::runtime_error("The layer named '" + label() + "' should be connected to one layer.");
  }
	
  // Compute the new values for this layer
  auto prev_itr = (*(_prevs.begin()))->begin();
  for(auto &v: _values){ 
    v = _f(*prev_itr);
    ++prev_itr;
  }
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::function::function(std::string function_name,
									      std::vector<int> shape,
									      std::string label) {
  std::shared_ptr<neuralfield::function::Layer> l;
  
  if(function_name == "sigmoid") {
    l = std::make_shared<neuralfield::function::VectorizedFunction>(label, [](double x) -> double { return 1.0 / (1.0 + exp(-x));}, shape);
  }
  else if(function_name == "relu") {
    l = std::make_shared<neuralfield::function::VectorizedFunction>(label, [](double x) -> double {
	if(x <= 0.0)
	  return 0.0;
	else
	  return x;
      }, shape);
  }
  else {
    throw std::invalid_argument(std::string("Unknown function : ") + function_name);
  }

  auto net = neuralfield::get_current_network();
  net += l;
  
  return l;
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::function::function(std::string function_name,
									      int size,
									      std::string label) {
  return function(function_name, {size}, label);
}
std::shared_ptr<neuralfield::function::Layer> neuralfield::function::function(std::string function_name,
									      int size1,
									      int size2,
									      std::string label) {
  return function(function_name, {size1, size2}, label);
}





neuralfield::function::Constant::Constant(std::string label,
					  std::vector<int> shape):
  neuralfield::function::Layer(label, 1, shape) {

}

void neuralfield::function::Constant::update() {
  
}

void neuralfield::function::Constant::set_parameters(std::vector<double> params) {
  
  neuralfield::function::Layer::set_parameters(params);
  for(auto& v: *this)
    v = _parameters[0];
	
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::function::constant(double value,
						       std::vector<int> shape,
						       std::string label) {
  auto l = std::make_shared<neuralfield::function::Constant>(neuralfield::function::Constant(label, shape));
  l->set_parameters({value});
  auto net = neuralfield::get_current_network();
  net += l;
  return l;
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::function::constant(double value,
						       int size,
						       std::string label) {
  return neuralfield::function::constant(value, {size}, label);
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::function::constant(double value,
						       int size1,
						       int size2,
						       std::string label) {
  return neuralfield::function::constant(value, {size1, size2}, label);
}
    
