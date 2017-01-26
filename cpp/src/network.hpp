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

      // We reorder the function layers in order to
      // evaluate them in the "correct order"
      // so that a layer at position i in the new collection
      // depends only on the function layers at position j < i
      std::list<std::shared_ptr<neuralfield::layer::FunctionLayer> > reordered_layers;
      bool at_least_one_insertion;
      while(_function_layers.size() != 0) {
	at_least_one_insertion = false;
	auto it = _function_layers.begin();
	auto it_r = std::back_inserter(reordered_layers);
	while(it != _function_layers.end()) {
	  if((*it)->has_all_dependencies_in(reordered_layers.begin(), reordered_layers.end())) {
	    *(it_r++) = *it;
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
	  std::cerr << "We still have layers for which we cannot determine the evaluation order, these are named : ";
	  for(auto l: _function_layers)
	    std::cerr << "\"" << l->label() << "\" ";
	  std::cerr << std::endl;
	  throw std::runtime_error("Remaining layers when trying to order their evaluation");
	}
      }

      _function_layers = reordered_layers;

      // std::cout << "Evaluation order of the function layers" << std::endl;
      // for(auto it: _function_layers)
      // 	std::cout << it->label() << std::endl;
      
      for(auto& l: _function_layers)
	l->propagate_values();
    }

    void step() {
      // The function layers are supposed to be loaded with their updated values
      // we therefore begin by evaluating all the buffered layers
      for(auto& l: _buffered_layers)
	l->update();
      // We expose the new values to the output
      for(auto& l: _buffered_layers)
	l->swap();

      // And then diffuse through the function layers
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
    net->_input_layers.push_back(l);
    net->register_labelled_layer(l);
      
    return net;
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
