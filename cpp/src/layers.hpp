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


#include "types.hpp"
#include "convolution_fftw.h"

namespace neuralfield {

  namespace layer {   

    class Layer {
    protected:
      std::string _label;
      parameters_type _parameters;
      values_type _values;
      values_type::size_type _size;
    public:
      Layer() = delete;
      
      Layer(std::string label,
	    typename parameters_type::size_type number_of_parameters,
	    typename values_type::size_type size):
	_label(label),
	_parameters(number_of_parameters),
	_values(size),
	_size(size) {
	std::fill(_values.begin(), _values.end(), 0.0);
      }
      
      Layer(const Layer&) = default;

      std::string label() {
	return _label;
      }
      
      void set_parameters(std::initializer_list<double> params) {
	assert(params.size() == _parameters.size());
	std::copy(params.begin(), params.end(), _parameters.begin());
      }   
      
      const values_type& values() const {
	return _values;
      }
      
      values_iterator begin() {
	return _values.begin();
      }
      
      values_iterator end() {
	return _values.end();
      }
      
      double operator()(values_type::size_type index) const{
	return _values[index];
      }
     void fill_values(values_type& ovalues) {
	assert(ovalues.size() == _values.size());
	std::copy(_values.begin(), _values.end(), ovalues.begin());
      }
      

      virtual void propagate_values() = 0;
      virtual void update() = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Layer& l) {
      for(auto& v: l.values())
	os << v << " ";
      return os;
    }

    

    class FunctionLayer: public Layer {
    protected:
      std::shared_ptr<Layer> _prev;
      
    public:
      FunctionLayer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    typename values_type::size_type size):
	Layer(label, number_of_parameters, size),
	_prev(nullptr) {
      }

      FunctionLayer(const FunctionLayer& other):
	Layer(other),
	_prev(other._prev) {
	
      }
      
      void connect(std::shared_ptr<Layer> prev) {
	_prev = prev;
      }

      void propagate_values(void) override {
      }
      
      void update(void) override {
        return;
      }
      
    };
    




    
    /*! \class AbstractInputLayer
     * @brief An AbstractInputLayer exists only to provide a base class because of the operator+= of the Network class
     * \sa example-001-basics.cpp
     */

    class AbstractInputLayer : public Layer {
    public:
      AbstractInputLayer(std::string label,
		 typename parameters_type::size_type number_of_parameters,
			 typename values_type::size_type size):
	Layer(label, number_of_parameters, size) {}

      void propagate_values(void) override {
      }
      
      void update(void) override {
      }
    };

    /*! \class InputLayer
     * @brief An input layer hosts some values and can be fed with some INPUT data with a user provided function.
     * \tparam INPUT The type of the input feeding the layer
     * \sa example-001-basics.cpp
     */
    template<typename INPUT>
    class InputLayer: public AbstractInputLayer {
    protected:
      
    public:
      using input_type = INPUT;

      //! The type of function mapping the input_type to the internal type values_type
      using fill_input_type = std::function<void (values_iterator, values_iterator, const input_type&) >;

      fill_input_type _fill_input; //!< A function for feeding a values_type from an INPUT

      InputLayer(std::string label,
		 typename parameters_type::size_type number_of_parameters,
		 typename values_type::size_type size,
		 fill_input_type fill_input) :
	AbstractInputLayer(label, number_of_parameters, size),
	_fill_input(fill_input) {}


      void propagate_values(void) override {
      }
      
      void update(void) override {
      }
      
      void fill(const input_type& input) {
	_fill_input(this->_values.begin(), this->_values.end(), input);
      }
    };


    template<typename INPUT>
    std::shared_ptr<AbstractInputLayer> input(values_type::size_type size, typename InputLayer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<InputLayer<INPUT> >(InputLayer<INPUT>(label, 0, size, fill_input));
    }


    class BufferedLayer : public Layer {

    protected:
      std::shared_ptr<neuralfield::layer::Layer> _prev;
      neuralfield::values_type _buffer;
    public:
      BufferedLayer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    typename values_type::size_type size):
	Layer(label, number_of_parameters, size),
	_prev(nullptr),
	_buffer(size) {
	std::fill(_buffer.begin(), _buffer.end(), 0.0);
      }

      void connect(std::shared_ptr<Layer> prev) {
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
      LeakyIntegrator(std::string label,
		      double alpha,
		      typename values_type::size_type size):
	BufferedLayer(label, 1, size) {
	_parameters[0] = alpha;
      }

      void update(void) override {
	auto prev_itr = _prev->begin();
	auto buffer_itr = _buffer.begin();
	double alpha = _parameters[0];
	for(auto& v: _values) {
	  (*buffer_itr) = (1. - alpha) * v + alpha * *prev_itr;
	  ++prev_itr;
	  ++buffer_itr;
	}
      }
      
    };


    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha, typename values_type::size_type size, std::string label = "") {
      return std::make_shared<LeakyIntegrator>(LeakyIntegrator(label, alpha, size));
    }

  }


  namespace function {

    
    class VectorizedFunction : public neuralfield::layer::FunctionLayer {
    protected:
      std::function<double(double)> _f;
    public:
      VectorizedFunction(std::string label,
			 std::function<double(double)> f,
			 typename values_type::size_type size):
	FunctionLayer(label, 0, size), _f(f) {
      }

      
      void propagate_values() override {
	// Ask the previous layer to update its values
	_prev->propagate_values();

	// And then compute the new values for this layer
	auto prev_itr = _prev->begin();
	for(auto &v: _values){ 
	  v = _f(*prev_itr);
	  ++prev_itr;
	}
      }
      
      void update(void) override {
	
      }
    };


    std::shared_ptr<neuralfield::layer::FunctionLayer> function(std::string function_name, typename values_type::size_type size, std::string label="") {
      if(function_name == "sigmoid") {
	return std::make_shared<VectorizedFunction>(label, [](double x) -> double { return 1.0 / (1.0 + exp(-x));}, size);
      }
      else if(function_name == "relu") {
	return std::make_shared<VectorizedFunction>(label, [](double x) -> double {
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
      FFTW_Convolution::Workspace ws;
      bool _toric;
      double * kernel;

    private:

      void init_convolution() {
	FFTW_Convolution::clear_workspace(ws);
	delete[] kernel;
	kernel = 0;

	int k_size;
	int k_center;
	std::function<int(int, int)> dist;
	if(_toric) {
	  k_size = _size;
	  k_center = 0;
	  FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, 1, _size, 1, k_size);
	  
	  dist = [this] (int x, int y) {
	    return std::min(abs(x-y), int(this->_size) - abs(x - y));
	  };
	  
	}
	else {
	  k_size = 2*_size-1;
	  k_center = k_size/2;
	  FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME, 1, _size, 1, k_size);
	  
	  dist = [] (int x, int y) {
	    return abs(x-y);
	  };
	  
	}
	
	kernel = new double[k_size];
	double A = _parameters[0];
	double s = _parameters[1];
	for(int i = 0 ; i < k_size ; ++i) {
	  double d = dist(i, k_center);
	  kernel[i] = A * exp(-d*d / (2.0 * s*s));
	}
      }
      
    public:
      Gaussian(std::string label, double A, double s, bool toric, typename values_type::size_type size):
	neuralfield::layer::FunctionLayer(label, 2, size),
	  _toric(toric),
	  kernel(0) {
	_parameters[0] = A;
	_parameters[1] = s;
	init_convolution();
      }

      ~Gaussian() {
	FFTW_Convolution::clear_workspace(ws);
	delete[] kernel;
      }
      
      void propagate_values() override {
	// Ask the previous layer to update its values
	_prev->propagate_values();

	// And then compute the new values for this layer
	auto prev_itr = _prev->begin();
	for(auto &v: _values){ 
	  v = _parameters[0] * (*prev_itr);
	  ++prev_itr;
	}
      }
      
      void update(void) override {
        return;
      }
    };
      
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric, typename values_type::size_type size, std::string label="") {
      return std::shared_ptr<Gaussian>(new Gaussian(label, A, s, toric, size));
    }

    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(typename values_type::size_type size, std::string label="") {
      return std::shared_ptr<Gaussian>(new Gaussian(label, 0.0, 0.1, true, size));
    }
  
  }
}
