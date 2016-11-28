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

namespace neuralfield {

	namespace layer {


		class ValuesLayer1D;
		std::ostream& operator<< (std::ostream& os, const ValuesLayer1D& l);

		/*! \class ValuesLayer
		 * @brief A layer hosting some values.
		 * A layer hosting some values. It provides the basis from which the InputLayer inherits
		 */
		class ValuesLayer1D {

			protected:
				values_type _values;

			public:
				ValuesLayer1D() = delete;
				ValuesLayer1D(typename values_type::size_type size): _values(size) {}
				ValuesLayer1D(const ValuesLayer1D&) = default;

				ValuesLayer1D& operator=(const ValuesLayer1D& other) = default;
				friend std::ostream& operator<< (std::ostream& os, const ValuesLayer1D& l);
				values_iterator begin() {
					return _values.begin();
				}
				values_iterator end() {
					return _values.end();
				}
				double operator()(values_type::size_type index) const{
					return _values[index];
				}

		};

		std::ostream& operator<<(std::ostream& os, const ValuesLayer1D& l) {
			for(auto& v: l._values)
				os << v << " ";
			return os;
		}

		ValuesLayer1D values(typename values_type::size_type size) {
			return ValuesLayer1D(size);
		}


		class ParametricLayer {
			protected:
				parameters_type _parameters;
			public:
				ParametricLayer() = delete;
				ParametricLayer(typename parameters_type::size_type size): _parameters(size) {}
				ParametricLayer(const ParametricLayer&) = default;
				ParametricLayer& operator=(const ParametricLayer& other) = default;
				void set_parameters(std::initializer_list<double> params) {
					assert(params.size() == _parameters.size());
					std::copy(params.begin(), params.end(), _parameters.begin());
				}

		};



		/*! \class InputLayer
		 * @brief An input layer hosts some values and can be fed with some INPUT data with a user provided function
		 * \tparam INPUT The type of the input feeding the layer
		 * \sa example-001-basics.cpp
		 */
		template<typename INPUT, typename BASE>
			class InputLayer: public BASE {
				public:
					using input_type = INPUT;

					//! The type of function mapping the input_type to the internal type values_type
					using fill_input_type = std::function<void (values_iterator, values_iterator, const input_type&) >;

					fill_input_type _fill_input; //!< A function for feeding a values_type from an INPUT

					InputLayer(typename values_type::size_type size, fill_input_type fill_input) : BASE(size),
					_fill_input(fill_input) {}


					void fill(const input_type& input) {
						_fill_input(this->_values.begin(), this->_values.end(), input);
					}
			};


		template<typename INPUT>
			InputLayer<INPUT, ValuesLayer1D> input(values_type::size_type size, typename InputLayer<INPUT, ValuesLayer1D>::fill_input_type fill_input) {
				return InputLayer<INPUT, ValuesLayer1D>(size, fill_input);
			}


		/*
		   template<typename INPUT>
		   InputLayer<INPUT> input(values_type::size_type width, values_type::size_type height, typename InputLayer<INPUT>::fill_input_type fill_input) {
		   return InputLayer<INPUT>(size, fill_input);
		   }
		   */

		/*! \class LinkType
		 * @brief a type allowing to specialize the linklayers in order to optimally
		 * computes its contribution.
		 */
		/*
		   enum class LinkType: int {
		   naive,
		   step,
		   fft
		   } LinkType;
		   */
		/*! \class LinkLayer
		 * @brief A link layer allows to connect ValuesLayer together
		 * @Todo: do we need to host the result of the contribution ???? 
		 */
			class LinkLayer1D: public ParametricLayer {
protected:
	ValuesLayer1D* _prev;

				public:
					LinkLayer1D(parameters_type::size_type number_of_parameters): ParametricLayer(number_of_parameters), _prev(nullptr) {}
					//! A link layer must receive inputs from a ValueLayer
					void connect(const ValuesLayer1D& ) {}	
					virtual void evaluate(ValuesLayer1D&) = 0;
			};

		class GaussianLink1D : public LinkLayer1D {
			public:
				GaussianLink1D() : LinkLayer1D(2) {}
				void evaluate(ValuesLayer1D&) override final {}
		};

		GaussianLink1D gaussian_link1D(void) {
			return GaussianLink1D();
		}

		GaussianLink1D gaussian_link(const ValuesLayer1D& input) {
			GaussianLink1D g;
			g.connect(input);
			return g;
		}
	}


}
