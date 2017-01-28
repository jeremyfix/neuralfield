#include "layers.hpp"

neuralfield::layer::Layer::Layer(std::string label,
			  typename neuralfield::parameters_type::size_type number_of_parameters,
			  std::vector<int> shape):
  _label(label),
  _parameters(number_of_parameters),
  _shape(shape) {

  _size = 1;       
  for(auto s: _shape)
    _size *= s;
  
  _values.resize(_size);
  
  std::fill(_values.begin(), _values.end(), 0.0);
  }
      

unsigned int neuralfield::layer::Layer::size() const {
  return _size;
}

std::vector<int> neuralfield::layer::Layer::shape() const {
  return _shape;
}
      
std::string neuralfield::layer::Layer::label() {
  return _label;
}
      
void neuralfield::layer::Layer::set_parameters(std::initializer_list<double> params) {
  assert(params.size() == _parameters.size());
  std::copy(params.begin(), params.end(), _parameters.begin());
}   
      
neuralfield::values_iterator neuralfield::layer::Layer::begin() {
  return _values.begin();
}
      
neuralfield::values_iterator neuralfield::layer::Layer::end() {
  return _values.end();
}

neuralfield::values_const_iterator neuralfield::layer::Layer::begin() const{
  return _values.begin();
}
      
neuralfield::values_const_iterator neuralfield::layer::Layer::end() const {
  return _values.end();
}


std::ostream& neuralfield::layer::operator<<(std::ostream& os, const neuralfield::layer::Layer& l) {
  for(auto const& v: l)
    os << v << " ";
  return os;
}










 



