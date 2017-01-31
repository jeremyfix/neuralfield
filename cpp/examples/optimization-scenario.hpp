#include "neuralfield.hpp"

using Input = std::vector<double>;

class Scenario {
protected:
  unsigned int _nb_steps;
  std::vector<int> _shape;
  int _size;
  Input input;
public:
  Scenario(unsigned int nb_steps,
	   std::vector<int> shape) :
    _nb_steps(nb_steps),
    _shape(shape) {
    _size = 1;
    for(auto s: shape)
      _size *= s;
    input.resize(_size);
  }

  virtual double evaluate(std::shared_ptr<neuralfield::Network> net) = 0;
};


enum class CompetitionType {Random, Structured};

template<CompetitionType T>
class CompetitionScenario : public Scenario {

private:
  void generate_input();
public:

  CompetitionScenario(unsigned int nb_steps,
		      std::vector<int> shape) :
    Scenario(nb_steps, shape) {
  }

  std::vector<double> local_argmax(void) {
    if(_shape.size() == 1) {
      return {0.};
    }
    else if(_shape.size() == 2) {
      return {0., 0.};
    }
    else {
      throw std::runtime_error("Cannot compute local argmax in dimensions higher than 2");
    }
  }

  std::vector<double> get_lower_bound(std::vector<double> max_pos) {
  }

  std::vector<double> get_upper_bound(std::vector<double> max_pos) {
  }
  
  double evaluate(std::shared_ptr<neuralfield::Network> net) override {
    // For a competition scenario, there must be a single bump
    // its location depends on the argmax of a convoluted input
    generate_input();
    
    net->reset();
    
    net->set_input<Input>("input", input);

    for(unsigned int i = 0 ; i < _nb_steps; ++i)
      net->step();
    auto fu = net->get("fu");
    
    // We now evaluate the fitness
    auto max_pos = local_argmax();

    // Then we build up the templates
    auto lb = get_lower_bound(max_pos);
    auto ub = get_upper_bound(max_pos);

    
    std::vector<double>::iterator it_lb, it_ub;
    it_lb = lb.begin();
    it_ub = ub.begin();
    double f = 0.0;
    for(auto& v: *fu) {
      if(v < *it_lb)
	f += (v-*it_lb)*(v-*it_lb);
      else if(v > *it_ub)
	f += (v-*it_ub)*(v-*it_ub);
      ++it_lb;
      ++it_ub;
    }
    return f;
  }

};

template<>
void CompetitionScenario<CompetitionType::Random>::generate_input() {

}

template<>
void CompetitionScenario<CompetitionType::Structured>::generate_input() {

}


