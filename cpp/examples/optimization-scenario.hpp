#include "neuralfield.hpp"
#include <fstream>

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
  std::vector<double> _lb;
  std::vector<double> _ub;
  double _sigma;
  double _dsigma;
  void generate_input();
  
public:

  CompetitionScenario(unsigned int nb_steps,
		      std::vector<int> shape,
		      double sigma,
		      double dsigma) :
    Scenario(nb_steps, shape),
    _lb(_size),
    _ub(_size),
    _sigma(sigma),
    _dsigma(dsigma) {
    if(shape.size() != 1 && shape.size() != 2)
      throw std::runtime_error("Cannot build competition scenario in dimensions higher than 2");
  }

  std::vector<double> local_argmax(void) {
    if(_shape.size() == 1) {
      return {0.};
    }
    else if(_shape.size() == 2) {
      return {0., 0.};
    }
    else 
      throw std::runtime_error("Cannot compute local argmax in dimensions higher than 2");
  }

  void fill_lower_bound(std::vector<double> max_pos) {
    // circular_rectified_cosine
    if(_shape.size() == 1) {
      std::cout << "fill " << std::endl;
      int i = 0;
      int N = _shape[0];
      double d;
      double cx = max_pos[0];
      double s = _sigma - _dsigma;
      for(auto& v: _lb) {
	d = std::min(fabs(cx - i), N - fabs(cx - i));
	v = std::cos(M_PI/4.0 * d / s);
	if(d >= 2*s)
	  v = 0;
	else
	  v = std::cos(M_PI/4.0 * d / s);
	++i;
      }
    }
    else if(_shape.size() == 2) {
    }
    else
      throw std::runtime_error("Cannot compute lb in dimensions higher than 2");
  }

  void fill_upper_bound(std::vector<double> max_pos) {
    // sigmoid gaussian
    auto f = [](double x) {
      return 1.0 / (1.0 + exp(-15. * (x - 0.5)));
    };
    if(_shape.size() == 1) {
      double s = _sigma + _dsigma;
      auto g = [s](double d) {
	return exp(-d*d/(2.0 * s * s));
      };
      int N = _shape[0];
      int i = 0;
      double cx = max_pos[0];
      double d;
      for(auto& v: _ub) {
	d = std::min(fabs(cx - i), N - fabs(cx - i));
	v = f(g(d));
	++i;
      }
    }
    else if(_shape.size() == 2) {
    }
    else
      throw std::runtime_error("Cannot compute ub in dimensions higher than 2");
  }

  void dump_bounds() {
    std::ofstream out;
    out.open("bounds.data");

    auto it_lb = _lb.begin();
    auto it_ub = _ub.begin();
    
    for(unsigned int i = 0 ; i < _size; ++i, ++it_lb, ++it_ub) 
      out << i << " " << *it_lb << " " << *it_ub << std::endl;

    out.close();  
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
    fill_lower_bound(max_pos);
    fill_upper_bound(max_pos);

    
    std::vector<double>::iterator it_lb, it_ub;
    it_lb = _lb.begin();
    it_ub = _ub.begin();
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


