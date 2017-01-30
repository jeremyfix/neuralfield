#include "neuralfield.hpp"


class Scenario {  
public:
  Scenario() {
  }

  virtual double evaluate(std::shared_ptr<neuralfield::Network> net) = 0;
};

class CompetitionScenario : public Scenario {
public:
  double evaluate(std::shared_ptr<neuralfield::Network> net) override {
    return 0.;
  }
};
