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
  
  std::cout << "Setting size of input " << _size << std::endl;
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










std::shared_ptr<neuralfield::layer::FunctionLayer> neuralfield::link::gaussian(double A, double s, bool toric,
									       std::initializer_list<int> shape,
									       std::string label) {
  return std::shared_ptr<neuralfield::link::Gaussian>(new neuralfield::link::Gaussian(label, A, s, toric, shape));
}
std::shared_ptr<neuralfield::layer::FunctionLayer> neuralfield::link::gaussian(double A, double s, bool toric,
									       int size,
									       std::string label) {
  return neuralfield::link::gaussian(A, s, toric, {size}, label);
}
std::shared_ptr<neuralfield::layer::FunctionLayer> neuralfield::link::gaussian(double A, double s, bool toric,
									       int size1, int size2,
									       std::string label) {
  return neuralfield::link::gaussian(A, s, toric, {size1, size2}, label);
}    


neuralfield::link::SumLayer::SumLayer(std::string label,
				      std::shared_ptr<neuralfield::layer::Layer> l1,
				      std::shared_ptr<neuralfield::layer::Layer> l2):
  neuralfield::layer::FunctionLayer(label, 0, l1->shape()){
  assert(l1->shape() == l2->shape());
  connect(l1);
  connect(l2);
}

void neuralfield::link::SumLayer::update(void) {
  auto it_self = begin();
  auto it_prevs = _prevs.begin();
	
  auto it1 = (*it_prevs++)->begin();
  auto it2 = (*it_prevs++)->begin();
  while(it_self != end())
    *it_self = (*it1++) + (*it2++);
}

std::shared_ptr<neuralfield::layer::FunctionLayer> neuralfield::operator+(std::shared_ptr<neuralfield::layer::FunctionLayer> l1,
									  std::shared_ptr<neuralfield::layer::FunctionLayer> l2) {
  std::string label("");
  if(l1->label() != "" && l2->label() != "")
    label = l1->label() + "+" + l2->label();
  return std::make_shared<neuralfield::link::SumLayer>(label, l1, l2);
}
