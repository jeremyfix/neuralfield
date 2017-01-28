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

#include <list>
#include <map>
#include <memory>

#include "types.hpp"
#include "layers.hpp"
#include "input_layers.hpp"
#include "function_layers.hpp"
#include "buffered_layers.hpp"

namespace neuralfield {

  class Network;
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::input::AbstractLayer> l);
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::function::Layer> l);
  std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::buffered::Layer> l);
  
  std::shared_ptr<Network> network();
  
  class Network {
  private:
    std::list<std::shared_ptr<neuralfield::input::AbstractLayer> > _input_layers;
    std::list<std::shared_ptr<neuralfield::function::Layer> > _function_layers;
    std::list<std::shared_ptr<neuralfield::buffered::Layer> > _buffered_layers;
    
    std::map<std::string, std::shared_ptr<neuralfield::layer::Layer>> _labelled_layers;
    
    void register_labelled_layer(std::shared_ptr<neuralfield::layer::Layer> layer);
    
  public:
    Network();

    void init();
    void step();

    std::shared_ptr<neuralfield::layer::Layer> get(std::string label);

    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::input::AbstractLayer> l);
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::function::Layer> l);
    friend std::shared_ptr<Network> operator+=(std::shared_ptr<Network> net, std::shared_ptr<neuralfield::buffered::Layer> l);
  };

}
