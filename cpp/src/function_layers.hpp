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
#include <vector>
#include <functional>

#include "layers.hpp"
#include "types.hpp"

namespace neuralfield {

  namespace function {
    
    class Layer: public neuralfield::layer::Layer {
    protected:
      std::list<std::shared_ptr<neuralfield::layer::Layer> > _prevs;
      
    public:
      Layer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    std::vector<int> shape);

      Layer(const Layer& other);

      void connect(std::shared_ptr<neuralfield::layer::Layer> prev);
      bool is_connected();
      void update(void) override;
      bool can_be_evaluated_from(const std::map<std::shared_ptr<neuralfield::layer::Layer>, bool>& evaluation_status);
    };

    class VectorizedFunction : public neuralfield::function::Layer {
    protected:
      std::function<double(double)> _f;
    public:
      VectorizedFunction(std::string label,
			 std::function<double(double)> f,
			 std::vector<int> shape);

      void update() override;
    };

    std::shared_ptr<neuralfield::function::Layer> function(std::string function_name,
								std::vector<int> shape,
								std::string label="");

    std::shared_ptr<neuralfield::function::Layer> function(std::string function_name,
								int size,
								std::string label="");
    std::shared_ptr<neuralfield::function::Layer> function(std::string function_name,
								int size1,
								int size2,
								std::string label="");


    class Constant : public neuralfield::function::Layer {
    public:
      Constant(std::string label,
	       std::vector<int> shape);
      void update() override;
      void set_parameters(std::vector<double> params) override;
      
    };

    std::shared_ptr<neuralfield::function::Layer> constant(double value,
							   std::vector<int> shape,
							   std::string label="");

    std::shared_ptr<neuralfield::function::Layer> constant(double value,
							   int size,
							   std::string label="");

    std::shared_ptr<neuralfield::function::Layer> constant(double value,
							   int size1,
							   int size2,
							   std::string label="");


    class UniformNoise : public  neuralfield::function::Layer {
    private:
      double _min, _max;
    public:
      UniformNoise(std::string label,
	       std::vector<int> shape,
	       double min, double max );
      void update() override;
      
    };

    std::shared_ptr<neuralfield::function::Layer> uniform_noise(double min, double max,
								std::vector<int> shape,
								std::string label="");

    std::shared_ptr<neuralfield::function::Layer> uniform_noise(double min, double max,
								int size,
								std::string label="");

    std::shared_ptr<neuralfield::function::Layer> uniform_noise(double min, double max,
								int size1,
								int size2,
								std::string label="");
    
  }
}
