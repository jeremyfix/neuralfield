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


#include <functional>
#include <iostream>
#include <string>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <list>
#include <map>

#include "types.hpp"
#include "convolution_fftw.h"

namespace neuralfield {

  namespace function {
    class Layer;
  }
  namespace link {
    class SumLayer;
  }
  
  namespace layer {   

    class Layer;
    std::shared_ptr<neuralfield::function::Layer> operator+(std::shared_ptr<neuralfield::layer::Layer> l1,
							    std::shared_ptr<neuralfield::layer::Layer> l2);
    
    class Layer {
    protected:
      std::string _label;
      parameters_type _parameters;
      std::vector<int> _shape;
      values_type::size_type _size;
      values_type _values;
    public:
      Layer() = delete;
      
      Layer(std::string label,
	    typename parameters_type::size_type number_of_parameters,
	    std::vector<int> shape);
      
      Layer(const Layer&) = default;

      unsigned int size() const;
      std::vector<int> shape() const;
      std::string label();
      
      void set_parameters(std::initializer_list<double> params);
      
      values_iterator begin();
      values_iterator end();
      values_const_iterator begin() const;
      values_const_iterator end() const;

      void reset();
      virtual void update() = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Layer& l);

  }
}
