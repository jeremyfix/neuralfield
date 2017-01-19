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

/*! \mainpage neuralfield : C++ dynamic neural field implementation
 *
 * This library provides several tools to build up various neural fields
 * A neural field is built up from a stack of layers which can be evaluated synchronously or asynchronously
 *
 * @example example-001-basics.cpp
 */


#include <types.hpp>
#include <tools.hpp>
#include <layers.hpp>
//#include <transfer_functions.hpp>
#include <integrator.hpp>
#include <network.hpp>

