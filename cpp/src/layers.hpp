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


#include "types.hpp"
#include "convolution_fftw.h"

namespace neuralfield {

  namespace layer {   

    class Layer {
    protected:
      std::string _label;
      parameters_type _parameters;
      std::list<int> _shape;
      values_type::size_type _size;
      values_type _values;
    public:
      Layer() = delete;
      
      Layer(std::string label,
	    typename parameters_type::size_type number_of_parameters,
	    std::list<int> shape):
	_label(label),
	_parameters(number_of_parameters),
	_shape(shape) {

	_size = 1;       
	for(auto s: _shape)
	  _size *= s;
	
	_values.resize(_size);
	
	std::fill(_values.begin(), _values.end(), 0.0);
      }
      
      Layer(const Layer&) = default;

      unsigned int size() const {
	return _size;
      }
      
      std::string label() {
	return _label;
      }
      
      void set_parameters(std::initializer_list<double> params) {
	assert(params.size() == _parameters.size());
	std::copy(params.begin(), params.end(), _parameters.begin());
      }   
      
      values_iterator begin() {
	return _values.begin();
      }
      
      values_iterator end() {
	return _values.end();
      }

      values_const_iterator begin() const{
	return _values.begin();
      }
      
      values_const_iterator end() const {
	return _values.end();
      }
      
      /*
      double operator()(values_type::size_type index) const{
	return _values[index];
      }
      */
      void fill_values(values_type& ovalues) {
	assert(ovalues.size() == _values.size());
	std::copy(_values.begin(), _values.end(), ovalues.begin());
      }
      

      virtual void propagate_values() = 0;
      virtual void update() = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Layer& l) {
      for(auto const& v: l)
	os << v << " ";
      return os;
    }

    

    class FunctionLayer: public Layer {
    protected:
      std::shared_ptr<Layer> _prev;
      
    public:
      FunctionLayer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    std::list<int> shape):
	Layer(label, number_of_parameters, shape),
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

      bool has_all_dependencies_in(std::list<std::shared_ptr<FunctionLayer>>::iterator begin,
				   std::list<std::shared_ptr<FunctionLayer>>::iterator end) {
	auto ptr = std::dynamic_pointer_cast<neuralfield::layer::FunctionLayer>(_prev);
	if(ptr) {
	  // _prev is a FunctionLayer
	  // we then check if this layer is already in the evaluated layers [begin; end[
	  return std::find(begin, end, _prev) != end;
	}
	else
	  return true;
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
			 std::list<int> shape):
	Layer(label, number_of_parameters, shape) {}

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
		 const std::list<int>& shape,
		 fill_input_type fill_input) :
	AbstractInputLayer(label, number_of_parameters, shape),
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
    std::shared_ptr<AbstractInputLayer> input(std::initializer_list<int> shape, typename InputLayer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<InputLayer<INPUT> >(InputLayer<INPUT>(label, 0, shape, fill_input));
    }
    
    // Utilitary function for building up 1D InputLayer
    template<typename INPUT>
    std::shared_ptr<AbstractInputLayer> input(int size, typename InputLayer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<InputLayer<INPUT> >(InputLayer<INPUT>(label, 0, {size}, fill_input));
    }

    // Utilitary function for building up 2D InputLayer
    template<typename INPUT>
    std::shared_ptr<AbstractInputLayer> input(int size1, int size2, typename InputLayer<INPUT>::fill_input_type fill_input, std::string label="") {
      return std::make_shared<InputLayer<INPUT> >(InputLayer<INPUT>(label, 0, {size1, size2}, fill_input));
    }

    class BufferedLayer : public Layer {

    protected:
      std::shared_ptr<neuralfield::layer::Layer> _prev;
      neuralfield::values_type _buffer;
    public:
      BufferedLayer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    std::list<int> shape):
	Layer(label, number_of_parameters, shape),
	_prev(nullptr) {
	_buffer.resize(this->size());
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
		      std::list<int> shape):
	BufferedLayer(label, 1, shape) {
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


    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      std::initializer_list<int> shape,
						      std::string label = "") {
      return std::make_shared<LeakyIntegrator>(LeakyIntegrator(label, alpha, shape));
    }
    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      int size,
						      std::string label = "") {
      return std::make_shared<LeakyIntegrator>(LeakyIntegrator(label, alpha, {size}));
    }
    std::shared_ptr<LeakyIntegrator> leaky_integrator(double alpha,
						      int size1,
						      int size2,
						      std::string label = "") {
      return std::make_shared<LeakyIntegrator>(LeakyIntegrator(label, alpha, {size1, size2}));
    }

  }


  namespace function {

    
    class VectorizedFunction : public neuralfield::layer::FunctionLayer {
    protected:
      std::function<double(double)> _f;
    public:
      VectorizedFunction(std::string label,
			 std::function<double(double)> f,
			 std::list<int> shape):
	FunctionLayer(label, 0, shape), _f(f) {
      }

      
      void propagate_values() override {
	// Compute the new values for this layer
	auto prev_itr = _prev->begin();
	for(auto &v: _values){ 
	  v = _f(*prev_itr);
	  ++prev_itr;
	}
      }
      
      void update(void) override {
	
      }
    };


    std::shared_ptr<neuralfield::layer::FunctionLayer> function(std::string function_name,
								std::initializer_list<int> shape,
								std::string label="") {
      if(function_name == "sigmoid") {
	return std::make_shared<VectorizedFunction>(label, [](double x) -> double { return 1.0 / (1.0 + exp(-x));}, shape);
      }
      else if(function_name == "relu") {
	return std::make_shared<VectorizedFunction>(label, [](double x) -> double {
	    if(x <= 0.0)
	      return 0.0;
	    else
	      return x;
	  }, shape);
      }
      else {
	throw std::invalid_argument(std::string("Unknown function : ") + function_name);
      }
    }

    std::shared_ptr<neuralfield::layer::FunctionLayer> function(std::string function_name,
								int size,
								std::string label="") {
      return function(function_name, {size}, label);
    }
        std::shared_ptr<neuralfield::layer::FunctionLayer> function(std::string function_name,
								    int size1,
								    int size2,
								    std::string label="") {
	  return function(function_name, {size1, size2}, label);
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
      Gaussian(std::string label,
	       double A, double s, bool toric,
	       std::list<int> shape):
	neuralfield::layer::FunctionLayer(label, 2, shape),
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
	//FFTW_Convolution::convolve(ws, src, kernel);
	std::copy(ws.dst, ws.dst + _size, _values.begin());
      }
      
      void update(void) override {
        return;
      }
    };
      
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								std::initializer_list<int> shape,
								std::string label="") {
      return std::shared_ptr<Gaussian>(new Gaussian(label, A, s, toric, shape));
    }
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								int size,
								std::string label="") {
      return gaussian(A, s, toric, {size}, label);
    }
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								int size1, int size2,
								std::string label="") {
      return gaussian(A, s, toric, {size1, size2}, label);
    }    

    

  
  }
}
