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


class CompetitionScenario : public Scenario {

protected:
  double _sigma;
  double _dsigma;
  bool _toric;
  double* kernel;
  double *src;
  FFTW_Convolution::Workspace ws;
  
protected:
  virtual void generate_input() = 0;
  
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

      // Linear convolution
      k_shape = 2*_shape[0]-1;
      k_center = k_shape/2;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);
	  
      dist = [] (int x_src, int x_dst) {
	return fabs(x_src-x_dst);
      };

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
      std::vector<int> k_shape;
      std::vector<int> k_center;
      std::function<double(int, int, int, int)> dist;

      // Linear convolution
      k_shape.push_back(2*_shape[0]-1);
      k_shape.push_back(2*_shape[1]-1);
      k_center.push_back(k_shape[0]/2);
      k_center.push_back(k_shape[1]/2);

      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], _shape[1], k_shape[0], k_shape[1]);
	  
      dist = [] (int x_src, int y_src, int x_dst, int y_dst) {
	double dx = fabs(x_src-x_dst);
	double dy = fabs(y_src-y_dst);
	return sqrt(dx*dx + dy*dy);
      };

      int k_size = k_shape[0] * k_shape[1];
      kernel = new double[k_size];
      double * kptr = kernel;
      double A = 1.0;
      double s = _sigma;
      for(int i = 0 ; i < k_shape[0] ; ++i) 
	for(int j = 0 ; j < k_shape[1]; ++j, ++kptr) {
	  double d = dist(i, j, k_center[0], k_center[1]);
	  *kptr = A * exp(-d*d / (2.0 * s*s));
	}
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

      // We convolve the input
      std::copy(begin, end, src);
      FFTW_Convolution::convolve(ws, src, kernel);
      
      // And pick up the argmax
      std::vector<double> argmax = {0., 0.};
      double dstmax = ws.dst[0];
      double* dst = ws.dst;
      for(int i = 0 ; i < _shape[0]; ++i)
	for(int j = 0 ; j < _shape[1]; ++j, ++dst) {
	  if(*dst > dstmax) {
	    argmax[0] = i;
	    argmax[1] = j;
	    dstmax = *dst;
	  }
	}
      return argmax;
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

	if(d >= 2*s)
	  v = 0;
	else
	  v = std::cos(M_PI/4.0 * d / s);
	++i;
      }
    }
    else if(_shape.size() == 2) {
      double s = _sigma - _dsigma;
      auto it_lb = _lb.begin();

      std::function<double(int, int)> dist;
      if(_toric)
	dist = [this, max_pos] (int x_src, int y_src) {
	  double dx = std::min(fabs(max_pos[0] - x_src), this->_shape[0] - fabs(max_pos[0] - x_src));
	  double dy = std::min(fabs(max_pos[1] - y_src), this->_shape[1] - fabs(max_pos[1] - y_src));
	  return sqrt(dx*dx + dy*dy);
	};
      else
	dist = [max_pos] (int x_src, int y_src) {
	  double dx = fabs(max_pos[0] - x_src);
	  double dy = fabs(max_pos[1] - y_src);
	  return sqrt(dx*dx+dy*dy);
	};

      double d;
      for(int i = 0 ; i < _shape[0] ; ++i) {
	for(int j = 0 ; j < _shape[1]; ++j, ++it_lb) {
	  d = dist(i, j);

	  if(d >= 2*s)
	    *it_lb = 0;
	  else
	    *it_lb = std::cos(M_PI/4.0 * d / s);
	}
      }
      
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
      double s = _sigma + _dsigma;
      auto g = [s](double d) {
	return exp(-d*d/(2.0 * s * s));
      };
      
      std::function<double(int, int)> dist;
      if(_toric)
	dist = [this, max_pos] (int x_src, int y_src) {
	  double dx = std::min(fabs(max_pos[0] - x_src), this->_shape[0] - fabs(max_pos[0] - x_src));
	  double dy = std::min(fabs(max_pos[1] - y_src), this->_shape[1] - fabs(max_pos[1] - y_src));
	  return sqrt(dx*dx + dy*dy);
	};
      else
	dist = [max_pos] (int x_src, int y_src) {
	  double dx = fabs(max_pos[0] - x_src);
	  double dy = fabs(max_pos[1] - y_src);
	  return sqrt(dx*dx+dy*dy);
	};
      
      double d;
      auto it_ub = _ub.begin();
      for(int i = 0 ; i < _shape[0] ; ++i) {
	for(int j = 0 ; j < _shape[1]; ++j, ++it_ub) {
	  d = dist(i, j);
	  *it_ub = f(g(d));
	}
      }
      
    }
    else
      throw std::runtime_error("Cannot compute ub in dimensions higher than 2");
  }

  void dump_bounds() {

    auto it_lb = _lb.begin();
    auto it_ub = _ub.begin();

    if(_shape.size() == 1) {
      std::ofstream out_lb, out_ub;
      out_lb.open("lb_bound.data");
      out_ub.open("ub_bound.data");
      for(int i = 0 ; i < _shape[0]; ++i, ++it_lb, ++it_ub) {
	out_lb << *it_lb << std::endl;
	out_ub << *it_ub << std::endl;
      }
      out_lb.close();
      out_ub.close();
    }
    else if(_shape.size() == 2) {
      std::ofstream out_lb, out_ub;
      out_lb.open("lb_bound.data");
      out_ub.open("ub_bound.data");
      for(int i = 0 ; i < _shape[0] ; ++i) {
	for(int j = 0 ; j < _shape[1]; ++j, ++it_lb, ++it_ub) {
	  out_lb << *it_lb << std::endl;
	  out_ub << *it_ub << std::endl;
	}
	out_lb << std::endl;
	out_ub << std::endl;
      }
      out_lb.close();
      out_ub.close();
    }

    std::cout << "Bounds saved in lb_bound.data, ub_bound.data" << std::endl;
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

class RandomCompetition : public CompetitionScenario {
public:
  RandomCompetition(unsigned int nb_steps,
		    std::vector<int> shape,
		    double sigma,
		    double dsigma,
		    bool toric):
    CompetitionScenario(nb_steps, shape, sigma, dsigma, toric) {}

  void generate_input() {
    for(auto& v: _input)
      v = neuralfield::random::uniform(0., 1.);
  }
};

class StructuredCompetition : public CompetitionScenario {
private:
  int _nb_gaussians;
  double _sigma_gaussians;
  
public:
  StructuredCompetition(unsigned int nb_steps,
			std::vector<int> shape,
			double sigma,
			double dsigma,
			bool toric,
			int nb_gaussians,
			double sigma_gaussians):
    CompetitionScenario(nb_steps, shape, sigma, dsigma, toric),
    _nb_gaussians(nb_gaussians),
    _sigma_gaussians(sigma_gaussians) {}


  void add_gaussian_input(std::vector<double> center, double A, double sigma) {
    assert(shape.size() == center.size());
    if(_shape.size() == 1) {

      std::function<double(int)> dist;
      if(_toric)
	dist = [this, center] (int x_src) {
	  double dx = std::min(fabs(center[0] - x_src), this->_shape[0] - fabs(center[0] - x_src));
	  return dx;
	};
      else
	dist = [center] (int x_src) {
	  double dx = fabs(center[0] - x_src);
	  return dx;
	};
      

      auto it_input = _input.begin();
      for(int i = 0 ; i < _shape[0]; ++i, ++it_input) {
	double d = dist(i);
	*it_input += A*exp(-d*d/(2.0 * sigma * sigma));
      }
    }
    else if(_shape.size() == 2) {
      std::function<double(int, int)> dist;
      
      if(_toric)
	dist = [this, center] (int x_src, int y_src) {
	  double dx = std::min(fabs(center[0] - x_src), this->_shape[0] - fabs(center[0] - x_src));
	  double dy = std::min(fabs(center[1] - y_src), this->_shape[1] - fabs(center[1] - y_src));
	  return sqrt(dx*dx + dy*dy);
	};
      else
	dist = [center] (int x_src, int y_src) {
	  double dx = fabs(center[0] - x_src);
	  double dy = fabs(center[1] - y_src);
	  return sqrt(dx*dx+dy*dy);
	};

      auto it_input = _input.begin();
      for(int i = 0 ; i < _shape[0]; ++i) {
	for(int j = 0 ; j < _shape[1]; ++j, ++it_input) {
	  double d = dist(i, j);
	  *it_input += A*exp(-d*d/(2.0 * sigma * sigma));
	}
      }

      
    }

  }


  void generate_input() {
    // Reset the input
    std::fill(_input.begin(), _input.end(), 0.0);

    // Add the gaussians
    std::vector<double> center(_shape.size());
    for(unsigned int i = 0 ; i < _shape.size(); ++i)
      center[i] = neuralfield::random::uniform(0, _shape[i]-1);
    
    for(int i = 0 ; i < _nb_gaussians; ++i) {
      double A = neuralfield::random::uniform(0., 1.);
      add_gaussian_input(center, A, _sigma_gaussians);
    }

    // Normalize so that the input peaks at 1.
    double vmax = *(_input.begin());
    
    for(auto& v: _input) {
      if(v > vmax)
	vmax = v;
    }
    for(auto& v: _input) 
      v = v / vmax;

  }
  
};


