#include <neuralfield.hpp>
#include <fstream>

using Input = std::vector<double>;

class Scenario {
protected:
  unsigned int _nb_steps;
  std::vector<int> _shape;
  int _size;
  Input _input;
public:
  Scenario(unsigned int nb_steps,
	   std::vector<int> shape) :
    _nb_steps(nb_steps),
    _shape(shape) {
    _size = 1;
    for(auto s: shape)
      _size *= s;
    _input.resize(_size);
  }

  virtual double evaluate(std::shared_ptr<neuralfield::Network> net) = 0;
};


enum class CompetitionType {Random, Structured};

template<CompetitionType T>
class CompetitionScenario : public Scenario {

private:
  double _sigma;
  double _dsigma;
  bool _toric;
  double* kernel;
  double *src;
  FFTW_Convolution::Workspace ws;
  
  void generate_input();
  
public:
  std::vector<double> _lb;
  std::vector<double> _ub;
  

  CompetitionScenario(unsigned int nb_steps,
		      std::vector<int> shape,
		      double sigma,
		      double dsigma,
		      bool toric) :
    Scenario(nb_steps, shape),
    _sigma(sigma),
    _dsigma(dsigma),
    _toric(toric),
    _lb(_size),
    _ub(_size) {
    if(shape.size() != 1 && shape.size() != 2)
      throw std::runtime_error("Cannot build competition scenario in dimensions higher than 2");

    src = new double[_size];
    kernel = new double[_size];
    if(shape.size() == 1) {

      int k_shape;
      int k_center;
      std::function<double(int, int)> dist;
      /*      
	      if(toric) {
	      k_shape = _shape[0];
	      k_center = 0;
	      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], 1, k_shape, 1);
	  
	      dist = [k_shape] (int x_src, int x_dst) {
	      int dx = std::min(abs(x_src-x_dst), k_shape - abs(x_src - x_dst));
	      return dx;
	      };
	  
	      }
	      else {
      */
      k_shape = 2*_shape[0]-1;
      k_center = k_shape/2;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);
	  
      dist = [] (int x_src, int x_dst) {
	return fabs(x_src-x_dst);
      };
      //}

      kernel = new double[k_shape];
      double * kptr = kernel;
      double A = 1.0;
      double s = _sigma;
      for(int i = 0 ; i < k_shape ; ++i, ++kptr) {
	double d = dist(i, k_center);
	*kptr = A * exp(-d*d / (2.0 * s*s));
      }
    }
    else if(shape.size() == 2) {
    }
  }

  ~CompetitionScenario() {
    delete[] src;
    delete[] kernel;
  }

  template<typename ITER>
  std::vector<double> local_argmax(ITER begin, ITER end) {
    if(_shape.size() == 1) {
      // We convolve the input
      std::copy(begin, end, src);
      FFTW_Convolution::convolve(ws, src, kernel);
      
      // And pick up the argmax
      int argmax = 0;
      double dstmax = ws.dst[0];
      double* dst = ws.dst;
      for(int i = 0 ; i < _shape[0]; ++i, ++dst)
	if(*dst > dstmax) {
	  argmax = i;
	  dstmax = *dst;
	}
      return {double(argmax)};
    }
    else if(_shape.size() == 2) {
      throw std::logic_error("unimplemented");
      return {0., 0.};
    }
    else 
      throw std::runtime_error("Cannot compute local argmax in dimensions higher than 2");
  }

  void fill_lower_bound(std::vector<double> max_pos) {
    // circular_rectified_cosine
    if(_shape.size() == 1) {
      int i = 0;
      int N = _shape[0];
      double d;
      double cx = max_pos[0];
      double s = _sigma - _dsigma;
      for(auto& v: _lb) {
	if(_toric)
	  d = std::min(fabs(cx - i), N - fabs(cx - i));
	else
	  d = fabs(cx-i);
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
	if(_toric)
	  d = std::min(fabs(cx - i), N - fabs(cx - i));
	else
	  d = fabs(cx - i);
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
    
    for(int i = 0 ; i < _size; ++i, ++it_lb, ++it_ub) 
      out << i << " " << *it_lb << " " << *it_ub << std::endl;

    out.close();  
  }

  void compute_bounds() {
    auto max_pos = local_argmax(_input.begin(), _input.end());

    // Then we build up the templates
    fill_lower_bound(max_pos);
    fill_upper_bound(max_pos);
  }
  
  double evaluate(std::shared_ptr<neuralfield::Network> net) override {
    // For a competition scenario, there must be a single bump
    // its location depends on the argmax of a convoluted input
    generate_input();
    
    net->reset();
    
    net->set_input<Input>("input", _input);

    for(unsigned int i = 0 ; i < _nb_steps; ++i)
      net->step();
    auto fu = net->get("fu");
    
    //// We now evaluate the fitness
    
    // We build up the templates
    compute_bounds();

    
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
  for(auto& v: _input)
    v = neuralfield::random::uniform(0., 1.);
}

template<>
void CompetitionScenario<CompetitionType::Structured>::generate_input() {
  throw std::logic_error("unimplemented");
}


