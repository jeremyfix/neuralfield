#include "link_layers.hpp"
#include "network.hpp"
#include "tools.hpp"

#include <stdexcept>
#include <numeric>

neuralfield::link::Full::Full(
        std::string label,
        double value, 
        std::vector<int> shape):
    neuralfield::function::Layer(label, 1, shape) {
    neuralfield::function::Layer::set_parameters({value});
}

void neuralfield::link::Full::update() {

    auto prev = *(_prevs.begin());
    double contrib = std::accumulate(prev->begin(), prev->end(), 0.0);

    contrib = contrib * (this->get_parameter(0) / (double)this->size());
    std::fill(this->begin(), this->end(), contrib);
}


std::shared_ptr<neuralfield::function::Layer> neuralfield::link::full(
        double value,
        std::vector<int> shape,
        std::string label
        ) {
    auto l = std::shared_ptr<neuralfield::link::Full>(new neuralfield::link::Full(label, value, shape));
    auto net = neuralfield::get_current_network();
    net += l;
    return l;
}


std::shared_ptr<neuralfield::function::Layer> neuralfield::link::full(
        double value,
        int size,
        std::string label
        ) {
    return neuralfield::link::full(value, {size}, label);
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::link::full(
        double value,
        int size1,
        int size2,
        std::string label
        ) {
    return neuralfield::link::full(value, {size1, size2}, label);
}


neuralfield::link::Heaviside::Heaviside(std::string label,
        double value, 
        double radius,
        bool toric,
        std::vector<int> shape): 
    neuralfield::function::Layer(label, 2, shape),
    _toric(toric) {
    neuralfield::function::Layer::set_parameters({value, radius});
    _integralImage = new double[this->size()];

    if(radius > 1.0)
        throw std::runtime_error(std::string("It does not make sense to use a radius > 1 in ")+std::string(__PRETTY_FUNCTION__));
    if(toric)
        throw std::runtime_error("Heaviside toric is not yet implemented");
}

neuralfield::link::Heaviside::~Heaviside(void) {
    delete[] _integralImage;
}

void neuralfield::link::Heaviside::update(void) {

    double factor = this->get_parameter(0) / ((double) this->size());
    double radius = this->get_parameter(1);

    if(this->shape().size() == 1) {
        // We compute the integral image
        double* intImgPtr = _integralImage;
        auto prev = *(_prevs.begin());
        double acu = 0;
        for(const auto& v_prev: *prev) {
            acu += v_prev;
            *intImgPtr = acu;
            ++intImgPtr;
        }

        // And then compute the weight contribution
        // by computing the difference of the right values
        // in the integral image

        if(this->_toric) {
        }
        else {
            int idx = 0;
            int idx_for_left = (int)(radius * this->shape()[0]);
            auto it_left  = _integralImage;
            auto it_right = _integralImage + std::min(idx_for_left, this->shape()[0] - 1);
            auto it_endm1 = _integralImage + this->size() - 1;
            double contrib = 0.0;

            for(auto& v: *this) {
                contrib = *it_right;
                if(idx > idx_for_left)
                    contrib = contrib - *it_left;

                v = factor * contrib;

                if(idx > idx_for_left)
                    ++it_left;
                if(it_right != it_endm1)
                    ++it_right;
                ++idx;
            }

        }
    }
    else if(this->shape().size() == 2) {
        throw std::runtime_error("neuralfield::link::Heaviside::update cannot handle neuralfield of dimensions >= 2");
    }
    else 
        throw std::runtime_error("neuralfield::link::Heaviside::update cannot handle neuralfield of dimensions > 2");
}



std::shared_ptr<neuralfield::function::Layer> neuralfield::link::heaviside(
        double value,
        double radius,
        bool toric,
        std::vector<int> shape,
        std::string label) {
    auto l = std::shared_ptr<neuralfield::link::Heaviside>(new neuralfield::link::Heaviside(label, value, radius, toric, shape));
    auto net = neuralfield::get_current_network();
    net += l;
    return l;
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::link::heaviside(
        double      value, 
        double      radius,
        bool        toric,
        int         size,
        std::string label) {
    return neuralfield::link::heaviside(value, radius, toric, std::vector<int>({size}), label);
}

std::shared_ptr<neuralfield::function::Layer> neuralfield::link::heaviside(
        double value, 
        double radius,
        bool toric,
        int size1,
        int size2,
        std::string label) {
    return neuralfield::link::heaviside(value, radius, toric, std::vector<int>({size1, size2}), label);
}


void neuralfield::link::Gaussian::init_convolution() {
    FFTW_Convolution::clear_workspace(ws);
    delete[] kernel;
    kernel = 0;

    if(_shape.size() == 1) {
        int k_shape;
        int k_center;

        if(_toric) {
            k_shape = _shape[0];
            k_center = 0;
            FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], 1, k_shape, 1);

        }
        else {
            k_shape = 2*_shape[0]-1;
            k_center = k_shape/2;
            FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);

        }

        auto dist = neuralfield::distances::make_euclidean_1D({k_shape,}, _toric);

        kernel = new double[k_shape];
        double * kptr = kernel;
        double A = this->get_parameter(0);
        double s = this->get_parameter(1);
        for(int i = 0 ; i < k_shape ; ++i, ++kptr) {
            float d = dist(i, k_center);
            *kptr = A * exp(-d*d / (2.0 * s*s))  * 1.0 / (k_shape);
        }

        /// Scaling of the weights
        // This is usefull to prevent border effects when the connections are not toric
        if(_toric || !_scale) 
            std::fill(_scaling_factors, _scaling_factors + _size, 1.);
        else {
            double max_sum_weights = 0.0;
            kptr = kernel + int((_shape[0]-1.)/2.);
            for(int i = 0 ; i < _shape[0] ; ++i, ++kptr)
                max_sum_weights += *kptr;

            for(int i = 0 ; i < _shape[0]; ++i) {
                double sum_weights = 0.0;
                kptr = kernel + i;
                for(int j = 0 ; j < _shape[0]; ++j, ++kptr)
                    sum_weights += *kptr;

                _scaling_factors[i] = max_sum_weights / sum_weights;	
            }
        }


    }
    else if(_shape.size() == 2) {
        std::array<int, 2> k_shape;
        std::array<double, 2> k_center;
        if(_toric) {
            k_shape[0] = _shape[0];
            k_shape[1] = _shape[1];
            k_center[0] = 0.;
            k_center[1] = 0.;
            FFTW_Convolution::init_workspace(ws, FFTW_Convolution::CIRCULAR_SAME, _shape[0], _shape[1], k_shape[0], k_shape[1]);
        }
        else {
            k_shape[0] = 2*_shape[0]-1;
            k_shape[1] = 2*_shape[1]-1;
            k_center[0] = k_shape[0]/2;
            k_center[1] = k_shape[1]/2;
            FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], _shape[1], k_shape[0], k_shape[1]);
        }

        auto dist = neuralfield::distances::make_euclidean_2D(k_shape, _toric);

        kernel = new double[k_shape[0]*k_shape[1]];
        double A = this->get_parameter(0);
        double s = this->get_parameter(1);
        double * kptr = kernel;
        for(int i = 0 ; i < k_shape[0] ; ++i) {
            for(int j = 0 ; j < k_shape[1]; ++j, ++kptr) {
                float d = dist({(double)i, (double)j}, k_center);
                *kptr = A * exp(-d*d / (2.0 * s*s)) * 1.0 / (k_shape[0] * k_shape[1]);
            }
        }

        /// Scaling of the weights
        // This might be usefull to prevent border effects when the connections are not toric
        if(_toric || !_scale) 
            std::fill(_scaling_factors, _scaling_factors + _size, 1.);
        else {
            double max_sum_weights = 0.0;

            for(int i = 0 ; i < _shape[1]; ++i) // do not use kptr, we skip some values..
                for(int j = 0 ; j < _shape[0] ; ++j)
                    max_sum_weights += kernel[(i+int((_shape[1]-1.)/2.))*k_shape[0] + (j+int((_shape[0]-1.)/2.))];

            for(int i = 0 ; i < _shape[1]; ++i) // do not use kptr, we skip some values..
                for(int j = 0 ; j < _shape[0] ; ++j) {
                    double sum_weights = 0.0;
                    for(int k = 0 ; k < _shape[1]; ++k) 
                        for(int l = 0 ; l < _shape[0]; ++l) 
                            sum_weights += kernel[(k+i) * k_shape[0] + (l+j)];
                    _scaling_factors[i*_shape[0] + j] = max_sum_weights / sum_weights;
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
        bool scale,
        std::vector<int> shape):
    neuralfield::function::Layer(label, 2, shape),
    _toric(toric),
    kernel(0),
    _scale(scale)
{
    src = new double[_size];
    _scaling_factors = new double[_size];
    this->set_parameters({A, s});
    init_convolution();
}

neuralfield::link::Gaussian::~Gaussian() {
    FFTW_Convolution::clear_workspace(ws);
    delete[] kernel;
    delete[] src;
    delete[] _scaling_factors;
}

void neuralfield::link::Gaussian::set_parameters(std::vector<double> params) {
    neuralfield::function::Layer::set_parameters(params);
    init_convolution();
}

void neuralfield::link::Gaussian::update() {
    if(_prevs.size() != 1) {
        throw std::runtime_error("The layer named '" + label() + "' should be connected to one layer.");
    }

    // Compute the new values for this layer
    auto prev = *(_prevs.begin());

    std::copy(prev->begin(), prev->end(), src);
    FFTW_Convolution::convolve(ws, src, kernel);

    if(!_scale) {
        std::copy(ws.dst, ws.dst + _size, _values.begin());
    }
    else {
        double* dst_ptr = ws.dst;
        double* it_s = _scaling_factors;
        for(auto& v: _values) {
            v = (*dst_ptr) * (*it_s);
            ++it_s;
            ++dst_ptr;
        }
    }
}      



std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
        double s,
        bool toric,
        bool scale,
        std::vector<int> shape,
        std::string label) {
    auto l = std::shared_ptr<neuralfield::link::Gaussian>(new neuralfield::link::Gaussian(label, A, s, toric, scale, shape));
    auto net = neuralfield::get_current_network();
    net += l;
    return l;
}
std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
        double s,
        bool toric,
        bool scale,
        int size,
        std::string label) {
    return neuralfield::link::gaussian(A, s, toric, scale, std::vector<int>({size}), label);
}
std::shared_ptr<neuralfield::function::Layer> neuralfield::link::gaussian(double A,
        double s,
        bool toric,
        bool scale,
        int size1,
        int size2,
        std::string label) {
    return neuralfield::link::gaussian(A, s, toric, scale, std::vector<int>({size1, size2}), label);
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
    auto l1 = *(it_prevs++);
    auto l2 = *(it_prevs++);

    auto it1 = l1->begin();
    auto it2 = l2->begin();
    while(it_self != end())
        *(it_self++) = (*it1++) + (*it2++);
}


