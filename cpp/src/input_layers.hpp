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

#include "layers.hpp"

namespace neuralfield {
  namespace input {
    
    /*! \class AbstractInputLayer
     * @brief An AbstractInputLayer exists only to provide a base class because of the operator+= of the Network class
     * \sa example-001-basics.cpp
     */
    class AbstractLayer : public neuralfield::layer::Layer {
    public:
      AbstractLayer(std::string label,
			 typename parameters_type::size_type number_of_parameters,
			 std::vector<int> shape);

      void update(void) override;
      
    };

    /*! \class Layer
     * @brief An input layer hosts some values and can be fed with some INPUT data with a user provided function.
     * \tparam INPUT The type of the input feeding the layer
     * \sa example-001-basics.cpp
     */
    template<typename INPUT>
    class Layer: public AbstractLayer {
    protected:
      
    public:
      using input_type = INPUT;

      //! The type of function mapping the input_type to the internal type values_type
      using fill_input_type = std::function<void (values_iterator, values_iterator, const input_type&) >;

      fill_input_type _fill_input; //!< A function for feeding a values_type from an INPUT

      Layer(std::string label,
		 typename parameters_type::size_type number_of_parameters,
		 std::vector<int> shape,
		 fill_input_type fill_input) :
	AbstractLayer(label, number_of_parameters, shape),
	_fill_input(fill_input) {}
      
      void fill(const input_type& input) {
	_fill_input(this->_values.begin(), this->_values.end(), input);
      }
    };


    template<typename INPUT>
    std::shared_ptr<AbstractLayer> input(std::initializer_list<int> shape, typename Layer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<Layer<INPUT> >(Layer<INPUT>(label, 0, shape, fill_input));
    }
    
    // Utilitary function for building up 1D Layer
    template<typename INPUT>
    std::shared_ptr<AbstractLayer> input(int size, typename Layer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<Layer<INPUT> >(Layer<INPUT>(label, 0, {size}, fill_input));
    }

    // Utilitary function for building up 2D Layer
    template<typename INPUT>
    std::shared_ptr<AbstractLayer> input(int size1, int size2, typename Layer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<Layer<INPUT> >(Layer<INPUT>(label, 0, {size1, size2}, fill_input));
    }
  }
}
