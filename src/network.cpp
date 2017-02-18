#include "network.hpp"

std::shared_ptr<neuralfield::Network> neuralfield::Network::current_network;

std::shared_ptr<neuralfield::Network> neuralfield::network() {
  neuralfield::Network::current_network = std::make_shared<neuralfield::Network>();
  return neuralfield::Network::current_network;
}

std::shared_ptr<neuralfield::Network> neuralfield::get_current_network() {
  if(! neuralfield::Network::current_network)
    network();

  return neuralfield::Network::current_network;
}

std::shared_ptr<neuralfield::Network> neuralfield::operator+=(std::shared_ptr<neuralfield::Network> net, std::shared_ptr<neuralfield::input::AbstractLayer> l) {
  net->_input_layers.push_back(l);
  net->register_labelled_layer(l); 
  return net;
}
  
std::shared_ptr<neuralfield::Network> neuralfield::operator+=(std::shared_ptr<neuralfield::Network> net, std::shared_ptr<neuralfield::function::Layer> l) {
  net->_function_layers.push_back(l);
  net->register_labelled_layer(l);
  return net;
}
std::shared_ptr<neuralfield::Network> neuralfield::operator+=(std::shared_ptr<neuralfield::Network> net, std::shared_ptr<neuralfield::buffered::Layer> l) {
  net->_buffered_layers.push_back(l);
  net->register_labelled_layer(l);
  return net;
}


void neuralfield::Network::register_labelled_layer(std::shared_ptr<neuralfield::layer::Layer> layer) {
  if(layer->label() != "") {
    auto it = _labelled_layers.find(layer->label());
    if(it != _labelled_layers.end())
      throw std::logic_error("Duplicate layer name " + layer->label() + ". The labels of the layers must be unique.");
    _labelled_layers[layer->label()] = layer;
  }
}

neuralfield::Network::Network() {
}

void neuralfield::Network::init() {

  // We reorder the function layers in order to
  // evaluate them in the "correct order"
  // so that a layer at position i in the new collection
  // depends only on the function layers at position j < i
  std::map<std::shared_ptr<neuralfield::layer::Layer>, bool> evaluation_status;
  for(auto l: _input_layers)
    evaluation_status[l] = true;
  for(auto l: _buffered_layers)
    evaluation_status[l] = true;
  for(auto l: _function_layers)
    evaluation_status[l] = false;
      
  std::list<std::shared_ptr<neuralfield::function::Layer> > reordered_layers;
  bool at_least_one_insertion;
      
  while(_function_layers.size() != 0) {
    at_least_one_insertion = false;

    auto it = _function_layers.begin();
    auto it_r = std::back_inserter(reordered_layers);
    while(it != _function_layers.end()) {
      if((*it)->can_be_evaluated_from(evaluation_status)) {
	// We insert this function layer in the reordered list
	*(it_r++) = *it;
	// Indicate it is evaluated
	evaluation_status[*it] = true;
	// And therefore drop it out from the layers to be updated
	it = _function_layers.erase(it);
	at_least_one_insertion = true;
      }
      else
	++it;
    }
    if(!at_least_one_insertion) {
      // We removed none of the layers
      // which certainly means the defined network
      // contains a cycle
      std::string msg;
      msg += "We still have layers for which we cannot determine the evaluation order, these are named : ";
      for(auto l: _function_layers)
	msg += "\"" + l->label() + "\" ";
      throw std::runtime_error(msg);
    }
  }

  _function_layers = reordered_layers;

  for(auto l: _function_layers)
    l->update();
}

void neuralfield::Network::reset() {
  for(auto l: _buffered_layers)
    l->reset();
  for(auto l: _function_layers)
    l->update();
}

void neuralfield::Network::step() {
  // The function layers are supposed to be loaded with their updated values
  // we therefore begin by evaluating all the buffered layers
  for(auto l: _buffered_layers)
    l->update();
  // We expose the new values to the output
  for(auto l: _buffered_layers)
    l->swap();

  // And then diffuse through the function layers
  for(auto l: _function_layers)
    l->update();
}

std::shared_ptr<neuralfield::layer::Layer> neuralfield::Network::get(std::string label) {
  auto it = _labelled_layers.find(label);
  if(it == _labelled_layers.end())
    throw  std::logic_error("Cannot find layer labeled " + label);
    
  return it->second;    
}

void neuralfield::Network::print(void) {
  std::cout << std::string(8, '*') << " Network " << std::endl;
  std::cout << "  " << _input_layers.size() << " input layers" << std::endl;
  for(auto l: _input_layers)
    std::cout << "     '" << l->label() << "'" << std::endl;
  std::cout << "  " << _function_layers.size() << " function layers " << std::endl;
  for(auto l: _function_layers)
    std::cout << "     '" << l->label() << "'" << std::endl;
  std::cout << "  " << _buffered_layers.size() << " buffered layers " << std::endl;
  for(auto l: _buffered_layers)
    std::cout << "     '" << l->label() << "'" << std::endl;

  std::cout << std::string(16, '*') << std::endl;
  
}
