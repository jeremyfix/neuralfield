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
#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>

#include "layers.hpp"
#include "function_layers.hpp"
#include "types.hpp"

namespace neuralfield {
  namespace link {


      class Constant : public neuralfield::function::Layer {

        public:
            Constant(std::string label,
                    double value,
                    std::vector<int> shape);
            ~Constant(void);
            void update() override;
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
							   std::string label= "");

     class Gaussian : public neuralfield::function::Layer {
      
    protected:
      FFTW_Convolution::Workspace ws;
      bool _toric;
      double * kernel;
      double * src;
      bool _scale;
      //double * _scaling_factors;

    private:
      void init_convolution();
      
    public:
      double * _scaling_factors;
      
      Gaussian(std::string label,
	       double A,
	       double s,
	       bool toric,
	       bool scale,
	       std::vector<int> shape);
      
      ~Gaussian();

      void set_parameters(std::vector<double> params) override;
      void update() override;  
    };

    

    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   bool scale,
							   std::vector<int> shape,
							   std::string label="");
    
    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   bool scale,
							   int size,
							   std::string label="");
    
    std::shared_ptr<neuralfield::function::Layer> gaussian(double A,
							   double s,
							   bool toric,
							   bool scale,
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
}
