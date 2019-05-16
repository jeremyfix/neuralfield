#include <iostream>
#include <cstdio>

#include "optimization-scenario.hpp"

int main(int argc, char* argv[]) {
    if(argc != 4 && argc != 5) {
        std::cerr << "Usage " << argv[0] << " sigma dsigma N <M>" << std::endl;
        std::cerr << "   sigma and dsigma must be given in normalized coordinates." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> shape;
    shape.push_back(std::atoi(argv[3]));
    if(argc == 3)
        shape.push_back(std::atoi(argv[4]));


    double sigma = std::atof(argv[1]);
    double dsigma = std::atof(argv[2]);
    RandomCompetition scenar(100, shape, sigma, dsigma, true);

    std::vector<double> max_pos;
    for(const auto s: shape)
        max_pos.push_back(0.5 * s);
    scenar.fill_lower_bound(max_pos);
    scenar.fill_upper_bound(max_pos);

    scenar.dump_bounds();

}
