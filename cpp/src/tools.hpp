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
}

