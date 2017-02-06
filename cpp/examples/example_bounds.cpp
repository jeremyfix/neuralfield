#include <iostream>
#include <cstdio>

#include "optimization-scenario.hpp"

int main(int argc, char* argv[]) {
  if(argc != 2 && argc != 3) {
    std::cerr << "Usage " << argv[0] << " N <M>" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<int> shape;
  shape.push_back(std::atoi(argv[1]));
  if(argc == 3)
    shape.push_back(std::atoi(argv[2]));


  double sigma = 2.0;
  double dsigma = 0.5;
  RandomCompetition scenar(100, shape, sigma, dsigma, true);

  std::vector<double> max_pos;
  for(const auto s: shape)
    max_pos.push_back(5.);
  scenar.fill_lower_bound(max_pos);
  scenar.fill_upper_bound(max_pos);

  scenar.dump_bounds();
  
}
