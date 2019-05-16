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

#include <random>
#include <cstdlib>
#include <functional>

namespace neuralfield {
  namespace random {

    /**
     * @return a random value in [min,max[
     */
    inline double uniform(double min,double max) {
      return min + (max-min)*(std::rand()/(RAND_MAX+1.0));
    }

    /**
     * @return a random integer in [0,max[
     */
    template<typename VALUE>
    inline VALUE uniform(VALUE max) {
      return (VALUE)(max*(std::rand()/(RAND_MAX+1.0)));
    }

    /**
     * @param p in [0,1]
     * @return true with the probability proba.
     */
    inline bool proba(double p) {
      return uniform(0,1)<p;
    }

    /**
     * @brief Draws a sample from a normal distribution
     * @param mu : mean
     * @param std : standard deviation
     */
    inline bool randn(double mu, double std) {
      return 0.0;
    }


  }

  namespace distances {

      std::function<double(double, double)> make_euclidean_1D(std::array<int, 1> shape, 
              bool toric) {
          if(toric)
              return [shape](double i0, double i1) -> double {
                  double d0  = fabs(i0 - i1);
                  double dd0 = std::min(d0 , shape[0] - d0) / double(shape[0]);
                  return dd0;
              };
          else
              return [shape](double i0, double i1) -> double {
                  double d0 = fabs(i0 - i1) / double(shape[0]);
                  return d0;
              };

      }

      std::function<double(std::array<double, 2>, std::array<double, 2>)> make_euclidean_2D(std::array<int, 2> shape,
              bool toric) {
          if(toric)
              return [shape](std::array<double, 2> pos1,
                             std::array<double, 2> pos2) -> double {
                  double d0    = abs(pos1[0] - pos2[0]);
                  double dd0 = std::min(d0, shape[0]-d0) / double(shape[0]);
                  double d1    = abs(pos1[1] - pos2[1]);
                  double dd1 = std::min(d1, shape[1]-d1) / double(shape[1]);
                  return sqrt(dd0*dd0 + dd1*dd1);
              };
          else
              return [shape](std::array<double, 2> pos1,
                             std::array<double, 2> pos2) -> double {
                  double d0    = abs(pos1[0] - pos2[0]) / double(shape[0]);
                  double d1    = abs(pos1[1] - pos2[1]) / double(shape[1]);
                  return sqrt(d0*d0 + d1*d1);
              };

      }

  }
}

