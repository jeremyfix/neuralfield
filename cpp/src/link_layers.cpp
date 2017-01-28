#include "link_layers.hpp"



void neuralfield::link::Gaussian::init_convolution() {
  FFTW_Convolution::clear_workspace(ws);
  delete[] kernel;
  kernel = 0;
  
  if(_shape.size() == 1) {
    int k_shape;
    int k_center;
    std::function<double(int, int)> dist;
	  
    if(_toric) {
      k_shape = _shape[0];
      k_center = 0;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], 1, k_shape, 1);
	  
      dist = [k_shape] (int x_src, int x_dst) {
	int dx = std::min(abs(x_src-x_dst), k_shape - abs(x_src - x_dst));
	return dx;
      };
	  
    }
    else {
      k_shape = 2*_shape[0]-1;
      k_center = k_shape/2;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);
	  
      dist = [] (int x_src, int x_dst) {
	return fabs(x_src-x_dst);
      };
    }

	  
    kernel = new double[k_shape];
    double * kptr = kernel;
    double A = _parameters[0];
    double s = _parameters[1];
    for(int i = 0 ; i < k_shape ; ++i, ++kptr) {
      double d = dist(i, k_center);
      *kptr = A * exp(-d*d / (2.0 * s*s));
    }
  }
  else if(_shape.size() == 2) {
    std::vector<int> k_shape(2);
    std::vector<int> k_center(2);
    std::function<double(int, int, int, int)> dist;
    if(_toric) {
      k_shape[0] = _shape[0];
      k_shape[1] = _shape[1];
      k_center[0] = 0;
      k_center[1] = 0;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], _shape[1], k_shape[0], k_shape[1]);
	  
      dist = [k_shape] (int x_src, int y_src, int x_dst, int y_dst) {
	int dx = std::min(abs(x_src-x_dst), k_shape[0] - abs(x_src - x_dst));
	int dy = std::min(abs(y_src-y_dst), k_shape[1] - abs(y_src - y_dst));
	return sqrt(dx*dx + dy*dy);
      };
	  
    }
    else {
      k_shape[0] = 2*_shape[0]-1;
      k_shape[1] = 2*_shape[1]-1;
      k_center[0] = k_shape[0]/2;
      k_center[1] = k_shape[1]/2;
      FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], _shape[1], k_shape[0], k_shape[1]);
	  
      dist = [] (int x_src, int y_src, int x_dst, int y_dst) {
	int dx = x_src-x_dst;
	int dy = y_src-y_dst;
	return sqrt(dx*dx + dy*dy);
      };
    }
	
    kernel = new double[k_shape[0]*k_shape[1]];
    double A = _parameters[0];
    double s = _parameters[1];
    double * kptr = kernel;
    for(int i = 0 ; i < k_shape[0] ; ++i) {
      for(int j = 0 ; j < k_shape[1]; ++j, ++kptr) {
	double d = dist(i, j, k_center[0], k_center[1]);
	*kptr = A * exp(-d*d / (2.0 * s*s));
      }
    }
  }
  else 
    throw std::runtime_error("I cannot handle convolution layers in dimension > 2");
}

neuralfield::link::Gaussian::Gaussian(std::string label,
				      double A,
				      double s,
				      bool toric,
				      std::vector<int> shape):
  neuralfield::function::Layer(label, 2, shape),
  _toric(toric),
  kernel(0)
{
  src = new double[_size];
  _parameters[0] = A;
  _parameters[1] = s;
  init_convolution();
	
}
      
neuralfield::link::Gaussian::~Gaussian() {
  FFTW_Convolution::clear_workspace(ws);
  delete[] kernel;
  delete[] src;
}
      
void neuralfield::link::Gaussian::update() {
  if(_prevs.size() != 1) {
    throw std::runtime_error("The layer named '" + label() + "' should be connected to one layer.");
  }
	
  // Compute the new values for this layer
  auto prev = *(_prevs.begin());
	
  std::copy(prev->begin(), prev->end(), src);
  FFTW_Convolution::convolve(ws, src, kernel);
  std::copy(ws.dst, ws.dst + _size, _values.begin());
}      
    


std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
									  double s,
									  bool toric,
									  std::initializer_list<int> shape,
									  std::string label) {
  return std::shared_ptr<neuralfield::link::Gaussian>(new neuralfield::link::Gaussian(label, A, s, toric, shape));
}
std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
									  double s,
									  bool toric,
									  int size,
									  std::string label) {
  return neuralfield::link::gaussian(A, s, toric, {size}, label);
}
std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
									       double s,
									       bool toric,
									       int size1,
									       int size2,
									       std::string label) {
  return neuralfield::link::gaussian(A, s, toric, {size1, size2}, label);
}   




neuralfield::link::SumLayer::SumLayer(std::string label,
				      std::shared_ptr<neuralfield::layer::Layer> l1,
				      std::shared_ptr<neuralfield::layer::Layer> l2):
  neuralfield::function::Layer(label, 0, l1->shape()){
  assert(l1->shape() == l2->shape());
  connect(l1);
  connect(l2);
}

void neuralfield::link::SumLayer::update(void) {
  auto it_self = begin();
  auto it_prevs = _prevs.begin();
	
  auto it1 = (*it_prevs++)->begin();
  auto it2 = (*it_prevs++)->begin();
  while(it_self != end())
    *it_self = (*it1++) + (*it2++);
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::operator+(std::shared_ptr<neuralfield::layer::Layer> l1,
								     std::shared_ptr<neuralfield::layer::Layer> l2) {
  std::string label("");
  if(l1->label() != "" && l2->label() != "")
    label = l1->label() + "+" + l2->label();
  return std::make_shared<neuralfield::link::SumLayer>(label, l1, l2);
}
