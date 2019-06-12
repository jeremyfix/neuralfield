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

template<typename BOUNDS_TYPE>
double get_param(const BOUNDS_TYPE& lbounds,
        const BOUNDS_TYPE& ubounds,
        const double * params,
        unsigned int idx) {
    return lbounds[idx] + (ubounds[idx] - lbounds[idx]) * params[idx];
}

template<typename BOUNDS_TYPE>
auto build_network(std::shared_ptr<neuralfield::Network> net,
        const BOUNDS_TYPE& lbounds,
        const BOUNDS_TYPE& ubounds,
        const double * params){

    // Set the parameters of the field
    // params = [dttau     h,  Ap,   sm, ka, ks]
    double dt_tau = get_param(lbounds, ubounds, params, 0);
    double h      = get_param(lbounds, ubounds, params, 1);
    double Ap     = get_param(lbounds, ubounds, params, 2);
    double sm     = get_param(lbounds, ubounds, params, 3);
    double ka     = get_param(lbounds, ubounds, params, 4);
    double ks     = get_param(lbounds, ubounds, params, 5);

    double Am = ka * Ap;
    double sp = ks * sm;

    net->get("gexc")->set_parameters({Ap, sp});
    net->get("ginh")->set_parameters({Am, sm});
    net->get("h")->set_parameters({h});
    net->get("u")->set_parameters({dt_tau});

    return net;
}

template<typename TNETWORK_BUILD>
double evaluate(unsigned int nb_steps,
        double sigma,
        double dsigma,
        std::vector<int> shape,
        TNETWORK_BUILD& builder,
        const double* params) {

    auto net = builder(params);

    bool toric_fitness = false;

    // Test the net on the different scenarii
    auto s1 = RandomCompetition(nb_steps, shape, sigma, dsigma, toric_fitness);
    double f1 = s1.evaluate(net);

    auto s2 = StructuredCompetition(nb_steps, shape, sigma, dsigma, toric_fitness, 5, 1./5.);
    double f2 = s2.evaluate(net);

    return f1 + f2;
}


template<typename TNETWORK_BUILD>
void test(unsigned int nb_steps,
        double sigma,
        double dsigma,
        std::vector<int> shape,
        TNETWORK_BUILD& builder,
        double * params) {
    std::cout << "Testing" << std::endl;

    auto net = builder(params);

    bool toric_fitness = true;
    auto s1 = RandomCompetition(nb_steps, shape, sigma, dsigma, toric_fitness);

    std::cout << "Fitnesses " << std::endl;
    for(unsigned int i = 0 ; i < 10 ; ++i) {
        auto f1 = s1.evaluate(net);
        std::cout << f1 << " ";
    }
    std::cout << std::endl;

    // Export the input/output and templates of the last run
    auto input = net->get("input");
    auto fu = net->get("fu");

    std::ofstream out_input, out_fu;
    out_input.open("input.data");
    out_fu.open("fu.data");

    auto it_input = input->begin();
    auto it_fu = fu->begin();

    if(shape.size() == 1) {
        for(int i = 0 ; i < shape[0] ; ++i, ++it_input, ++it_fu) {
            out_input << *it_input << std::endl;
            out_fu << *it_fu << std::endl;
        }
    }
    else if(shape.size() == 2) {
        for(int i = 0 ; i < shape[0] ; ++i) {
            for(int j = 0 ; j < shape[1]; ++j, ++it_input, ++it_fu) {
                out_input << *it_input << std::endl;
                out_fu << *it_fu << std::endl;
            }
            out_input << std::endl;
            out_fu << std::endl;
        }
    }

    out_fu.close();
    out_input.close();

    s1.dump_bounds();
    std::cout << "The input, fu are dumped in input.data and fu.data"  << std::endl;
    std::cout << "You can use gnuplot e.g. to plot them  :" << std::endl;
    if(shape.size() == 1) {
        std::cout << "     plot \"input.data\" w l , \"fu.data\" w l, \"lb_bound.data\" w l, \"ub_bound.data\" w l " << std::endl;
    }
    else if(shape.size() == 2) {
        std::cout << "     splot \"input.data\" w l , \"fu.data\" w l, \"lb_bound.data\" w l, \"ub_bound.data\" w l " << std::endl;
    }

}

int main(int argc, char * argv[]) {

    if(argc != 5 and argc != 6) {
        std::cerr << "Script to optimize a 2D neural field for a competition scenario" << std::endl;
        std::cerr << "Usage : " << argv[0] << " sigma dsigma toric N <M>" << std::endl;
        std::exit(-1);
    }

    RNG_GENERATOR::rng_srand();
    RNG_GENERATOR::rng_warm_up();

    double dt_tau = 0.01;
    double baseline = 0.0;
    double Ap = 1.5;
    double sp = 2.;
    double Am = -1.3;
    double sm = 10.;
    bool toric = std::atoi(argv[3]);
    unsigned int Nsteps = 100;

    double sigma = std::atof(argv[1]);
    double dsigma = std::atof(argv[2]);

    std::vector<int> shape;

    shape.push_back(std::atoi(argv[4]));
    if(argc == 6)
        shape.push_back(std::atoi(argv[5]));

    auto input = neuralfield::input::input<Input>(shape, fillInput, "input");
    auto h     = neuralfield::function::constant(baseline, shape, "h");
    auto u     = neuralfield::buffered::leaky_integrator(dt_tau, shape, "u");
    auto g_exc = neuralfield::link::gaussian(Ap, sp, toric, false, shape,"gexc");
    auto g_inh = neuralfield::link::gaussian(Am, sm, toric, false, shape, "ginh");
    auto fu    = neuralfield::function::function("sigmoid", shape, "fu");

    g_exc->connect(fu);
    g_inh->connect(fu);
    fu->connect(u);
    u->connect(g_exc + g_inh + input + h);

    auto net = neuralfield::get_current_network();
    net->print();

    net->init();

    // Parametrization of popot

    const unsigned int Nparams = 6;
    const unsigned int nb_evaluations = 1;

    //                                  dttau     h,  Ap,   sm, ka, ks 
    std::array<double, Nparams> tlbounds({0.01, -5.0, 0.01,  0.0001, -1., 0.001});
    std::array<double, Nparams> tubounds({0.20,  5.0, 1000.0,   3.0, -0.0001, 1.});
    auto network_builder = [&tlbounds, &tubounds, net](const double * params) {
        return build_network(net, tlbounds, tubounds, params);
    };
    
    auto lbound = [] (size_t index) -> double { return 0.0;};
    auto ubound = [] (size_t index) -> double { return 1.0;};

    auto stop =   [] (double fitness, int epoch) -> bool { return epoch >= 1000 || fitness <= 1e-10;};

    auto cost_function = [Nsteps, shape, sigma, dsigma, &network_builder] (TVector& pos) -> double { 
        return evaluate(Nsteps, sigma, dsigma, shape,  network_builder, pos.getValuesPtr());
    };

    auto algo = popot::algorithm::stochastic_montecarlo_spso2006(Nparams, 
            lbound, 
            ubound, 
            stop, 
            cost_function, 
            nb_evaluations);

    // We run the algorithm with verbosity
    algo->run(1);

    std::cout << "Best particle :" << algo->getBest() << std::endl;

    auto best_params = algo->getBest().getPosition().getValuesPtr();
    test(Nsteps, sigma, dsigma, shape, network_builder, best_params);

    dt_tau   = get_param(tlbounds, tubounds, best_params, 0);
    baseline = get_param(tlbounds, tubounds, best_params, 1);
    Ap       = get_param(tlbounds, tubounds, best_params, 2);
    sm       = get_param(tlbounds, tubounds, best_params, 3);
    double ka     = get_param(tlbounds, tubounds, best_params, 4);
    double ks     = get_param(tlbounds, tubounds, best_params, 5);
    Am       = Ap * ka;
    sp       = sm * ks;

    std::cout << "Parameters : " << std::endl;
    std::cout << "  dt_tau : "   << dt_tau   << std::endl;
    std::cout << "  h      : "   << baseline << std::endl;
    std::cout << "  Ap     : "   << Ap       << std::endl;
    std::cout << "  sp     : "   << sp       << std::endl;
    std::cout << "  Am     : "   << Am       << std::endl;
    std::cout << "  sm     : "   << sm       << std::endl;

    std::cout << std::endl;
    std::cout << " To test it : " << std::endl;
    std::cout << " ./examples/example-002-test "
        <<  dt_tau   << " "
        <<  baseline << " "
        <<  Ap       << " "
        <<  sp       << " "
        <<  Am       << " "
        <<  sm       << " "
        <<  int(toric) << " ";
    for(auto& s: shape)
        std::cout << s << " ";
    std::cout << std::endl;

    return 0;
}
