#include "neuralfield.hpp"
#include "optimization-scenario.hpp"
#include <iostream>

#include "rng_generators.h"
typedef popot::rng::CRNG RNG_GENERATOR;

#include "popot.h"
typedef popot::algorithm::ParticleStochasticSPSO::VECTOR_TYPE TVector;

void fillInput(neuralfield::values_iterator begin,
        neuralfield::values_iterator end,
        const Input& x) {
    std::copy(x.begin(), x.end(), begin);
}

auto build_network(int size) {

    std::vector<int> shape{size, size};

    double dt_tau = 0.01;
    bool toric = false;
    bool scale = false;

    auto input = neuralfield::input::input<Input>(shape, fillInput, "input");

    auto h = neuralfield::function::constant(0.0, shape, "h");
    auto u = neuralfield::buffered::leaky_integrator(dt_tau, shape, "u");
    auto g_exc = neuralfield::link::gaussian(0.0, 0.1, toric, scale, shape,"gexc");
    auto g_inh =  neuralfield::link::gaussian(0.0, 0.1, toric, scale, shape, "ginh");
    auto fu = neuralfield::function::function("sigmoid", shape, "fu");

    g_exc->connect(fu);
    g_inh->connect(fu);
    fu->connect(u);
    u->connect(g_exc + g_inh + input + h);

    auto net = neuralfield::get_current_network();
    return net;
}


int main(int argc, char * argv[]) {

    std::cout << "Script to optimize a 2D neural field for a tracking scenario " << std::endl; 
    std::cout << std::endl;

    auto small_net = build_network(20);   
    small_net->print();
    small_net->init();
    
    neuralfield::clear_current_network();

    auto big_net = build_network(40);   
    big_net->print();
    big_net->init();
}


