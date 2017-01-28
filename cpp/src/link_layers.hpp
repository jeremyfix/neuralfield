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
#include <initializer_list>
#include <algorithm>


#include "layers.hpp"
#include "function_layers.hpp"
#include "types.hpp"

namespace neuralfield {
  namespace link {

    class Gaussian : public neuralfield::function::Layer {
      
    protected:
      FFTW_Convolution::Workspace ws;
      bool _toric;
      double * kernel;
      double * src;

    private:
      void init_convolution();
      
    public:
      Gaussian(std::string label,
	       double A,
	       double s,
	       bool toric,
	       std::vector<int> shape);
      
      ~Gaussian();
      
      void update() override;  
    };

    

    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   std::initializer_list<int> shape,
							   std::string label="");
    
    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   int size,
							   std::string label="");
    
    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   int size1,
							   int size2,
							   std::string label= "");
      

    class SumLayer: public neuralfield::function::Layer {
      
    public:
      SumLayer(std::string label,
	       std::shared_ptr<neuralfield::layer::Layer> l1,
	       std::shared_ptr<neuralfield::layer::Layer> l2);
      
      void update(void) override;
    };
  }

  std::shared_ptr<neuralfield::function::Layer> operator+(std::shared_ptr<neuralfield::layer::Layer> l1,
							  std::shared_ptr<neuralfield::layer::Layer> l2);
}
