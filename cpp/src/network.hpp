#pragma once

#include "types.hpp"
#include "layers.hpp"

#include <list>
#include <map>
#include <memory>

namespace neuralfield {

  class Network;
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l);
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l);

  
  class Network {
  private:
    std::list<std::shared_ptr<neuralfield::layer::FunctionLayer>> _function_layers;
    std::list<std::shared_ptr<neuralfield::layer::BufferedLayer>> _buffered_layers;

    std::map<std::string, std::shared_ptr<neuralfield::layer::Layer>> _labelled_layers;
    
    void register_labelled_layer(std::shared_ptr<neuralfield::layer::Layer> layer) {
      if(layer->label()) {
	OICI !!!
      }
    }
  public:
    Network() {}

    void init() {
      for(auto& l: _function_layers)
	l->propagate_values();
    }

    void step() {
      for(auto& l: _buffered_layers)
	l->update();
      for(auto& l: _buffered_layers)
	l->swap();
      for(auto& l: _function_layers)
	l->propagate_values();
    }


    
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l);
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l);
  };

  std::shared_ptr<Network> network() {
    return std::make_shared<Network>();
  }

  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l) {
    net->_function_layers.push_back(l);
    net->register_labelled_layer(l);
      
    return net;
  }
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l) {
    net->_buffered_layers.push_back(l);
    net->register_labelled_layer(l);
    return net;
  }

}
