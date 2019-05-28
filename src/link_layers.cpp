#include "link_layers.hpp"
#include "network.hpp"
#include "tools.hpp"

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
        double A = _parameters[0];
        double s = _parameters[1];
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
        double A = _parameters[0];
        double s = _parameters[1];
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
    _parameters[0] = A;
    _parameters[1] = s;
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


