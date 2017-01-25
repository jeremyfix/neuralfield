#pragma once

/*
 *   Copyright (C) 2016,  CentraleSupelec
 *
 *   Author : Jeremy Fix
 *
 *   Contributor :
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU General Public
 *   License (GPL) as published by the Free Software Foundation; either
 *   version 3 of the License, or any later version.
 *   
 *   This library is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *   General Public License for more details.
 *   
 *   You should have received a copy of the GNU General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 *   Contact : jeremy.fix@centralesupelec.fr
 *
 */

#include "types.hpp"
#include "layers.hpp"

#include <list>
#include <map>
#include <memory>

namespace neuralfield {

  class Network;
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::AbstractInputLayer> l);
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l);
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l);
  
  class Network {
  private:
    std::list<std::shared_ptr<neuralfield::layer::AbstractInputLayer> > _input_layers;
    std::list<std::shared_ptr<neuralfield::layer::FunctionLayer> > _function_layers;
    std::list<std::shared_ptr<neuralfield::layer::BufferedLayer> > _buffered_layers;

    std::map<std::string, std::shared_ptr<neuralfield::layer::Layer>> _labelled_layers;
    
    void register_labelled_layer(std::shared_ptr<neuralfield::layer::Layer> layer) {
      if(layer->label() != "") {
        auto it = _labelled_layers.find(layer->label());
	if(it != _labelled_layers.end())
	  throw std::logic_error("Duplicate layer name " + layer->label() + ". The labels of the layers must be unique.");
	_labelled_layers[layer->label()] = layer;
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

    std::shared_ptr<neuralfield::layer::Layer> get(std::string label) {

    auto it = _labelled_layers.find(label);
    if(it == _labelled_layers.end())
      throw  std::logic_error("Cannot find layer labeled " + label);
    
    return it->second;
    
  }

    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::AbstractInputLayer> l);
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l);
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l);
  };


  
  std::shared_ptr<Network> network() {
    return std::make_shared<Network>();
  }

  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::AbstractInputLayer> l) {
    std::cout << "adding Input Layer" << std::endl;
    net->_input_layers.push_back(l);
    net->register_labelled_layer(l);
      
    return net;
  }
  
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::FunctionLayer> l) {
    std::cout << "adding function Layer" << std::endl;
    net->_function_layers.push_back(l);
    net->register_labelled_layer(l);
      
    return net;
  }
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::layer::BufferedLayer> l) {
    std::cout << "adding buffered Layer" << std::endl;
    net->_buffered_layers.push_back(l);
    net->register_labelled_layer(l);
    return net;
  }
}
