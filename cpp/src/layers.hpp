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

  namespace layer {   

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
      
      virtual void update() = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Layer& l);


    class FunctionLayer;
    std::shared_ptr<neuralfield::layer::FunctionLayer> operator+(std::shared_ptr<neuralfield::layer::FunctionLayer> l1,
								 std::shared_ptr<neuralfield::layer::FunctionLayer> l2);
    
    class FunctionLayer: public Layer {
    protected:
      std::list<std::shared_ptr<Layer> > _prevs;
      
    public:
      FunctionLayer(std::string label,
		    typename parameters_type::size_type number_of_parameters,
		    std::vector<int> shape):
	Layer(label, number_of_parameters, shape) {
      }

      FunctionLayer(const FunctionLayer& other):
	Layer(other),
	_prevs(other._prevs) {
	
      }
      
      void connect(std::shared_ptr<Layer> prev) {
	_prevs.push_back(prev);
      }

      bool is_connected() {
	bool connected = true;
	auto it = _prevs.begin();
	while(connected && it != _prevs.end()) {
	  connected &= bool(*it);
	  ++it;
	}
	return connected;
      }
      
      void update(void) override {
	if(! is_connected()) {
	  throw std::runtime_error("The layer named '" + label() + "' has some undefined previous layers.");
	}	
      }

      bool can_be_evaluated_from(const std::map<std::shared_ptr<Layer>, bool>& evaluation_status) {
	bool can_be_evaluated = true;
	auto it = _prevs.begin();
	while(can_be_evaluated && it != _prevs.end()) {
	  can_be_evaluated &= evaluation_status.at(*it);
	  ++it;
	}
	return can_be_evaluated;
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
			 std::vector<int> shape):
	Layer(label, number_of_parameters, shape) {}

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
		 std::vector<int> shape,
		 fill_input_type fill_input) :
	AbstractInputLayer(label, number_of_parameters, shape),
	_fill_input(fill_input) {}
      
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
		    std::vector<int> shape):
	Layer(label, number_of_parameters, shape),
	_prev(nullptr) {
	_buffer.resize(this->size());
	std::fill(_buffer.begin(), _buffer.end(), 0.0);
      }

      void connect(std::shared_ptr<Layer> prev) {
	_prev = prev;
      }

      bool is_connected() {
	return bool(_prev);
      }      

      void update(void) override {
	
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
		      std::vector<int> shape):
	BufferedLayer(label, 1, shape) {
	_parameters[0] = alpha;
      }

      void update(void) override {
	if(!_prev) {
	  throw std::runtime_error("The layer named '" + label() + "' has an undefined previous layer.");
	}
	
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
			 std::vector<int> shape):
	FunctionLayer(label, 0, shape), _f(f) {
      }

      
      void update() override {
	if(_prevs.size() != 1) {
	  throw std::runtime_error("The layer named '" + label() + "' should be connected to one layer.");
	}
	
	// Compute the new values for this layer
	auto prev_itr = (*(_prevs.begin()))->begin();
	for(auto &v: _values){ 
	  v = _f(*prev_itr);
	  ++prev_itr;
	}
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
      double * src;

    private:

      void init_convolution() {
	FFTW_Convolution::clear_workspace(ws);
	delete[] kernel;
	kernel = 0;

	if(_shape.size() == 1) {
	   int k_shape;
	  int k_center;
	  std::function<double(int, int)> dist;
	  
	  if(_toric) {
	    k_shape = _shape[0];
	    k_center = 0;
	    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], 1, k_shape, 1);
	  
	    dist = [k_shape] (int x_src, int x_dst) {
	      int dx = std::min(abs(x_src-x_dst), k_shape - abs(x_src - x_dst));
	      return dx;
	    };
	  
	  }
	  else {
	    k_shape = 2*_shape[0]-1;
	    k_center = k_shape/2;
	    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);
	  
	    dist = [] (int x_src, int x_dst) {
	      return fabs(x_src-x_dst);
	    };
	  }

	  
	  kernel = new double[k_shape];
	  double * kptr = kernel;
	  double A = _parameters[0];
	  double s = _parameters[1];
	  for(int i = 0 ; i < k_shape ; ++i, ++kptr) {
	    double d = dist(i, k_center);
	    *kptr = A * exp(-d*d / (2.0 * s*s));
	  }
	}
	else if(_shape.size() == 2) {
	  std::vector<int> k_shape(2);
	  std::vector<int> k_center(2);
	  std::function<double(int, int, int, int)> dist;
	  if(_toric) {
	    k_shape[0] = _shape[0];
	    k_shape[1] = _shape[1];
	    k_center[0] = 0;
	    k_center[1] = 0;
	    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], _shape[1], k_shape[0], k_shape[1]);
	  
	    dist = [k_shape] (int x_src, int y_src, int x_dst, int y_dst) {
	      int dx = std::min(abs(x_src-x_dst), k_shape[0] - abs(x_src - x_dst));
	      int dy = std::min(abs(y_src-y_dst), k_shape[1] - abs(y_src - y_dst));
	      return sqrt(dx*dx + dy*dy);
	    };
	  
	  }
	  else {
	    k_shape[0] = 2*_shape[0]-1;
	    k_shape[1] = 2*_shape[1]-1;
	    k_center[0] = k_shape[0]/2;
	    k_center[1] = k_shape[1]/2;
	    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], _shape[1], k_shape[0], k_shape[1]);
	  
	    dist = [] (int x_src, int y_src, int x_dst, int y_dst) {
	      int dx = x_src-x_dst;
	      int dy = y_src-y_dst;
	      return sqrt(dx*dx + dy*dy);
	    };
	  }
	
	  kernel = new double[k_shape[0]*k_shape[1]];
	  double A = _parameters[0];
	  double s = _parameters[1];
	  double * kptr = kernel;
	  for(int i = 0 ; i < k_shape[0] ; ++i) {
	    for(int j = 0 ; j < k_shape[1]; ++j, ++kptr) {
	      double d = dist(i, j, k_center[0], k_center[1]);
	      *kptr = A * exp(-d*d / (2.0 * s*s));
	    }
	  }
	 }
	else 
	  throw std::runtime_error("I cannot handle convolution layers in dimension > 2");
      }
      
    public:
      Gaussian(std::string label,
	       double A, double s, bool toric,
	       std::vector<int> shape):
	neuralfield::layer::FunctionLayer(label, 2, shape),
	_toric(toric),
	kernel(0)
      {
	src = new double[_size];
	_parameters[0] = A;
	_parameters[1] = s;
	init_convolution();
	
      }
      
      ~Gaussian() {
	FFTW_Convolution::clear_workspace(ws);
	delete[] kernel;
	delete[] src;
      }
      
      void update() override {
	if(_prevs.size() != 1) {
	  throw std::runtime_error("The layer named '" + label() + "' should be connected to one layer.");
	}
	
	// Compute the new values for this layer
	auto prev = *(_prevs.begin());
	
	std::copy(prev->begin(), prev->end(), src);
	FFTW_Convolution::convolve(ws, src, kernel);
	std::copy(ws.dst, ws.dst + _size, _values.begin());
      }      
    };
      
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								std::initializer_list<int> shape,
								std::string label="");
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								int size,
								std::string label="");
    std::shared_ptr<neuralfield::layer::FunctionLayer> gaussian(double A, double s, bool toric,
								int size1, int size2,
								std::string label="");

    
    
    class SumLayer: public neuralfield::layer::FunctionLayer {
    public:
      SumLayer(std::string label,
	       std::shared_ptr<Layer> l1,
	       std::shared_ptr<Layer> l2);
      void update(void) override;
    };
  

  }

  std::shared_ptr<neuralfield::layer::FunctionLayer> operator+(std::shared_ptr<neuralfield::layer::FunctionLayer> l1,
							       std::shared_ptr<neuralfield::layer::FunctionLayer> l2);
}
