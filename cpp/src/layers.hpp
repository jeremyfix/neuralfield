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

#include <types.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace neuralfield {

  namespace layer {   

    class Layer {
    protected:
      parameters_type _parameters;
    public:
      Layer(typename parameters_type::size_type number_of_parameters): _parameters(number_of_parameters){
      }

      virtual void propagate_values() = 0;
      virtual void update() = 0;
      virtual void fill_values(values_type& ovalues) = 0;
      
      void set_parameters(std::initializer_list<double> params) {
	assert(params.size() == _parameters.size());
	std::copy(params.begin(), params.end(), _parameters.begin());
      }
    };

    
    class ValuesLayer;
    std::ostream& operator<< (std::ostream& os, const ValuesLayer& l);

    /*! \class ValuesLayer
     * @brief A layer hosting some values.
     * A layer hosting some values. It provides the basis from which the InputLayer inherits
     */
    class ValuesLayer : public Layer {

    protected:
      std::shared_ptr<values_type> _values;

    public:
      ValuesLayer() = delete;
      ValuesLayer(typename parameters_type::size_type number_of_parameters,
		  typename values_type::size_type size):
	Layer(number_of_parameters),
	_values(std::make_shared<values_type>(size))
      {
	std::fill(_values->begin(), _values->end(), 0.0);
      }
      
      ValuesLayer(const ValuesLayer&) = default;

      ValuesLayer& operator=(const ValuesLayer& other) = default;
      friend std::ostream& operator<< (std::ostream& os, const ValuesLayer& l);

      values_type& values() {
	return *_values;
      }
      
      values_iterator begin() {
	return _values->begin();
      }
      values_iterator end() {
	return _values->end();
      }
      double operator()(values_type::size_type index) const{
	return (*_values)[index];
      }

      void fill_values(values_type& ovalues) override {
	assert(ovalues.size() == _values->size());
	std::copy(_values->begin(), _values->end(), ovalues.begin());
      }

    };

    std::ostream& operator<<(std::ostream& os, const ValuesLayer& l) {
      for(auto& v: *(l._values))
	os << v << " ";
      return os;
    }

    

    class FunctionLayer: public ValuesLayer {
    protected:
      std::shared_ptr<ValuesLayer> _prev;
      
    public:
      FunctionLayer(typename parameters_type::size_type number_of_parameters,
		    typename values_type::size_type size):
	ValuesLayer(number_of_parameters, size),  _prev(nullptr) {}
     
      void connect(std::shared_ptr<ValuesLayer> prev) {
	_prev = prev;
      }

    };
    




    
    /*! \class InputLayer
     * @brief An input layer hosts some values and can be fed with some INPUT data with a user provided function. It does not have any parameters
     * \tparam INPUT The type of the input feeding the layer
     * \sa example-001-basics.cpp
     */
    template<typename INPUT>
    class InputLayer: public ValuesLayer {
    protected:
      
    public:
      using input_type = INPUT;

      //! The type of function mapping the input_type to the internal type values_type
      using fill_input_type = std::function<void (values_iterator, values_iterator, const input_type&) >;

      fill_input_type _fill_input; //!< A function for feeding a values_type from an INPUT

      InputLayer(typename parameters_type::size_type number_of_parameters,
		 typename values_type::size_type size,
		 fill_input_type fill_input) : ValuesLayer(number_of_parameters, size),
					       _fill_input(fill_input) {}


      void propagate_values(void) override {
      }
      
      void update(void) override {
      }
      
      void fill(const input_type& input) {
	_fill_input(this->_values->begin(), this->_values->end(), input);
      }
    };


    template<typename INPUT>
    std::shared_ptr<InputLayer<INPUT> > input(values_type::size_type size, typename InputLayer<INPUT>::fill_input_type fill_input) {
      return std::make_shared<InputLayer<INPUT> >(InputLayer<INPUT>(0, size, fill_input));
    }


    class BufferedLayer : public ValuesLayer {

    protected:
      std::shared_ptr<neuralfield::layer::ValuesLayer> _prev;
      std::shared_ptr<neuralfield::values_type> _buffer;
    public:
      BufferedLayer(typename parameters_type::size_type number_of_parameters,
		    typename values_type::size_type size):
	ValuesLayer(number_of_parameters, size), _prev(nullptr), _buffer(std::make_shared<values_type>(size)) {
	std::fill(_buffer->begin(), _buffer->end(), 0.0);
      }

      void connect(std::shared_ptr<ValuesLayer> prev) {
	_prev = prev;
      }
      
      void propagate_values(void) override {
      }

      void swap(void) {
	std::swap(_buffer, _values);
      }
      
    };

    // u(t+1) = (1-alpha) * u(t) + alpha * i(t)
    class LeakyIntegrator : public BufferedLayer {
    public:
      LeakyIntegrator(double alpha, typename values_type::size_type size):
	BufferedLayer(1, size) {
	_parameters[0] = alpha;
      }

      void update(void) override {
	auto prev_itr = _prev->begin();
	auto buffer_itr = _buffer->begin();
	double alpha = _parameters[0];
	for(auto& v: values()) {
	  (*buffer_itr) = (1. - alpha) * v + alpha * *prev_itr;
	  ++prev_itr;
	  ++buffer_itr;
	}
      }
      
    };


    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha, typename values_type::size_type size) {
      return std::make_shared<LeakyIntegrator>(LeakyIntegrator(alpha, size));
    }

  }


  namespace function {


    
    
    class VectorizedFunction : public neuralfield::layer::FunctionLayer {
    protected:
      std::function<double(double)> _f;
    public:
      VectorizedFunction(std::function<double(double)> f,
			 typename values_type::size_type size):
	FunctionLayer(0, size), _f(f) {
      }

      
      void propagate_values() override {
	// Ask the previous layer to update its values
	_prev->propagate_values();

	// And then compute the new values for this layer
	auto prev_itr = _prev->begin();
	for(auto &v: values()){ 
	  v = _f(*prev_itr);
	  ++prev_itr;
	}
      }
      
      void update(void) override {
	
      }
    };


    std::shared_ptr<VectorizedFunction> function(std::string function_name, typename values_type::size_type size) {
      if(function_name == "sigmoid") {
	return std::make_shared<VectorizedFunction>([](double x) -> double { return 1.0 / (1.0 + exp(-x));}, size);
      }
      else if(function_name == "relu") {
	return std::make_shared<VectorizedFunction>([](double x) -> double {
	    if(x <= 0.0)
	      return 0.0;
	    else
	      return x;
	  }, size);
      }
      else {
	throw std::invalid_argument(std::string("Unknown function : ") + function_name);
      }
    }
    
  }


  namespace link {
    class Gaussian : public neuralfield::layer::FunctionLayer {
    protected:
    public:
      Gaussian(double Ap, double sp, typename values_type::size_type size):
	neuralfield::layer::FunctionLayer(2, size) {
	_parameters[0] = Ap;
	_parameters[1] = sp;
      }

      void propagate_values() override {
	// Ask the previous layer to update its values
	_prev->propagate_values();

	// And then compute the new values for this layer
	auto prev_itr = _prev->begin();
	for(auto &v: values()){ 
	  v = _parameters[0] * (*prev_itr);
	  ++prev_itr;
	}
      }
      
      void update(void) override {
        return;
      }
    };
      
    std::shared_ptr<Gaussian> gaussian(double Ap, double sp, typename values_type::size_type size) {
      return std::make_shared<Gaussian>(Ap, sp, size);
    }

    std::shared_ptr<Gaussian> gaussian(typename values_type::size_type size) {
      return std::make_shared<Gaussian>(0.0, 0.1, size);
    }
  
  }
}