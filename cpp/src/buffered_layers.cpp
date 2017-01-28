#include "buffered_layers.hpp"


neuralfield::buffered::Layer::Layer(std::string label,
				    typename parameters_type::size_type number_of_parameters,
				    std::vector<int> shape):
  neuralfield::layer::Layer(label, number_of_parameters, shape),
  _prev(nullptr) {
  _buffer.resize(this->size());
  std::fill(_buffer.begin(), _buffer.end(), 0.0);
  }

void neuralfield::buffered::Layer::connect(std::shared_ptr<neuralfield::layer::Layer> prev) {
  _prev = prev;
}

bool neuralfield::buffered::Layer::is_connected() {
  return bool(_prev);
}      

void neuralfield::buffered::Layer::update(void) {
	
}
      
void neuralfield::buffered::Layer::swap(void) {
  std::swap(_buffer, _values);
}


neuralfield::buffered::LeakyIntegrator::LeakyIntegrator(std::string label,
							double alpha,
							std::vector<int> shape):
  neuralfield::buffered::Layer(label, 1, shape) {
  _parameters[0] = alpha;
}

void neuralfield::buffered::LeakyIntegrator::update(void) {
  if(!_prev) {
    throw std::runtime_error("The layer named '" + label() + "' has an undefined previous layer.");
  }
	
  auto prev_itr = _prev->begin();
  auto buffer_itr = _buffer.begin();
  double alpha = _parameters[0];
  for(auto& v: _values) {
    (*buffer_itr) = (1. - alpha) * v + alpha * *prev_itr;
    ++prev_itr;
    ++buffer_itr;
  }
}


std::shared_ptr<neuralfield::buffered::LeakyIntegrator> neuralfield::buffered::leaky_integrator(double alpha,
												std::initializer_list<int> shape,
												std::string label) {
  return std::make_shared<neuralfield::buffered::LeakyIntegrator>(neuralfield::buffered::LeakyIntegrator(label, alpha, shape));
}
std::shared_ptr<neuralfield::buffered::LeakyIntegrator> neuralfield::buffered::leaky_integrator(double alpha,
												int size,
												std::string label) {
  return leaky_integrator(alpha, {size}, label);
}
std::shared_ptr<neuralfield::buffered::LeakyIntegrator> neuralfield::buffered::leaky_integrator(double alpha,
												int size1,
												int size2,
												std::string label) {
  return leaky_integrator(alpha, {size1, size2}, label);
}


