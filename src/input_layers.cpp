#include "input_layers.hpp"

neuralfield::input::AbstractLayer::AbstractLayer(std::string label,
						 typename parameters_type::size_type number_of_parameters,
						 std::vector<int> shape):
  neuralfield::layer::Layer(label, number_of_parameters, shape) {
}


void neuralfield::input::AbstractLayer::update(void) {
}
