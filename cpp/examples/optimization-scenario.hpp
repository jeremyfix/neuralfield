#include "neuralfield.hpp"

using Input = std::vector<double>;

class Scenario {
protected:
  unsigned int _nb_steps;
  std::vector<int> _shape;
  int _size;
public:
  Scenario(unsigned int nb_steps,
	   std::vector<int> shape) :
    _nb_steps(nb_steps),
    _shape(shape) {
    _size = 1;
    for(auto s: shape)
      _size *= s;
  }

  virtual double evaluate(std::shared_ptr<neuralfield::Network> net) = 0;
};


enum class CompetitionType {Random, Structured};

template<CompetitionType T>
class CompetitionScenario : public Scenario {
public:

  CompetitionScenario(unsigned int nb_steps,
		      std::vector<int> shape) :
    Scenario(nb_steps, shape) {
  }
  
  double evaluate(std::shared_ptr<neuralfield::Network> net) override ;

};

template<>
double CompetitionScenario<CompetitionType::Random>::evaluate(std::shared_ptr<neuralfield::Network> net) {
  return 0.;
}

template<>
double CompetitionScenario<CompetitionType::Structured>::evaluate(std::shared_ptr<neuralfield::Network> net) {
  return 0.;
}
/*
template<>
class CompetitionScenario<CompetitionType::Structured> {
  double evaluate(std::shared_ptr<neuralfield::Network> net) {
    return 0.;
  }
};
*/

