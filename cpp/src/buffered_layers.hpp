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

#include <memory>
#include <algorithm>
#include <vector>

#include "layers.hpp"
#include "types.hpp"

namespace neuralfield {

  namespace buffered {
    
    class Layer : public neuralfield::layer::Layer {

    protected:
      std::shared_ptr<neuralfield::layer::Layer> _prev;
      neuralfield::values_type _buffer;
    public:
      Layer(std::string label,
	    typename parameters_type::size_type number_of_parameters,
	    std::vector<int> shape);

      void connect(std::shared_ptr<neuralfield::layer::Layer> prev);

      bool is_connected();  

      void update(void) override;
      
      void swap(void);
      
    };

    // u(t+1) = (1-alpha) * u(t) + alpha * i(t)
    class LeakyIntegrator : public Layer {
    public:
      LeakyIntegrator(std::string label,
		      double alpha,
		      std::vector<int> shape);

      void update(void) override;
    };

    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      std::initializer_list<int> shape,
						      std::string label = "");
    
    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      int size,
						      std::string label = "");
    
    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      int size1,
						      int size2,
						      std::string label = "");

      
  }
}
