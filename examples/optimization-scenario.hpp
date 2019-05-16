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
                ////////////////////////////
                //  1D
                if(shape.size() == 1) {

                    int k_shape;
                    int k_center;

                    // Linear convolution
                    k_shape = 2*_shape[0]-1;
                    k_center = k_shape/2;
                    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], 1, k_shape, 1);

                    auto dist = neuralfield::distances::make_euclidean_1D({k_shape}, _toric);

                    kernel = new double[k_shape];
                    double * kptr = kernel;
                    double A = 1.0;
                    double s = _sigma;
                    for(int i = 0 ; i < k_shape ; ++i, ++kptr) {
                        double d = dist(i, k_center);
                        *kptr = A * exp(-d*d / (2.0 * s*s));
                    }
                }
                ////////////////////////////
                //  2D
                else if(shape.size() == 2) {
                    std::array<int, 2> k_shape;
                    std::array<double, 2> k_center;
                    
                    // Linear convolution
                    k_shape[0] = 2*_shape[0]-1;
                    k_shape[1] = 2*_shape[1]-1;
                    k_center[0] = k_shape[0]/2;
                    k_center[1] = k_shape[1]/2;

                    FFTW_Convolution::init_workspace(ws, FFTW_Convolution::LINEAR_SAME,  _shape[0], _shape[1], k_shape[0], k_shape[1]);


                    auto dist = neuralfield::distances::make_euclidean_2D(k_shape, false);

                    int k_size = k_shape[0] * k_shape[1];
                    kernel = new double[k_size];
                    double * kptr = kernel;
                    double A = 1.0;
                    double s = _sigma;
                    for(int i = 0 ; i < k_shape[0] ; ++i) 
                        for(int j = 0 ; j < k_shape[1]; ++j, ++kptr) {
                            double d = dist({double(i), double(j)}, k_center);
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
                ////////////////////////////
                //  1D
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
                ////////////////////////////
                //  2D
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
            ////////////////////////////
            //  1D
            if(_shape.size() == 1) {
                auto dist = neuralfield::distances::make_euclidean_1D({_shape[0]}, _toric);
                int i = 0;
                double d;
                double cx = max_pos[0];
                double s = _sigma - _dsigma;
                for(auto& v: _lb) {
                    d = dist(i, cx);

                    if(d >= 2*s)
                        v = 0;
                    else
                        v = std::cos(M_PI/4.0 * d / s);
                    ++i;
                }
            }
            ////////////////////////////
            //  2D
            else if(_shape.size() == 2) {
                double s = _sigma - _dsigma;
                auto it_lb = _lb.begin();

                auto dist = neuralfield::distances::make_euclidean_2D({_shape[0], _shape[1]}, _toric);

                double d;
                for(int i = 0 ; i < _shape[0] ; ++i) {
                    for(int j = 0 ; j < _shape[1]; ++j, ++it_lb) {
                        d = dist({double(i), double(j)}, {max_pos[0], max_pos[1]});

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
            ////////////////////////////
            //  1D
            if(_shape.size() == 1) {
                double s = _sigma + _dsigma;
                auto g = [s](double d) {
                    return exp(-d*d/(2.0 * s * s));
                };
                auto dist = neuralfield::distances::make_euclidean_1D({_shape[0]}, _toric);

                int i = 0;
                double cx = max_pos[0];
                double d;
                for(auto& v: _ub) {
                    d = dist(i, cx);
                    v = f(g(d));
                    ++i;
                }
            }
            ////////////////////////////
            //  2D
            else if(_shape.size() == 2) {
                double s = _sigma + _dsigma;
                auto g = [s](double d) {
                    return exp(-d*d/(2.0 * s * s));
                };

                auto dist = neuralfield::distances::make_euclidean_2D({this->_shape[0], this->_shape[1]}, _toric);
                
                double d;
                auto it_ub = _ub.begin();
                for(int i = 0 ; i < _shape[0] ; ++i) {
                    for(int j = 0 ; j < _shape[1]; ++j, ++it_ub) {
                        d = dist({double(i), double(j)}, {max_pos[0], max_pos[1]});
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
            ////////////////////////////
            //  1D
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
            ////////////////////////////
            //  2D
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

        void set_input(std::shared_ptr<neuralfield::Network> net) {
            generate_input();

            net->reset();

            net->set_input<Input>("input", _input);
        }

        double evaluate(std::shared_ptr<neuralfield::Network> net) override {
            // For a competition scenario, there must be a single bump
            // its location depends on the argmax of a convoluted input
            set_input(net);

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
            assert(_shape.size() == center.size());
            ////////////////////////////
            //  1D
            if(_shape.size() == 1) {
                auto dist = neuralfield::distances::make_euclidean_1D({_shape[0]}, _toric);
                auto it_input = _input.begin();
                for(int i = 0 ; i < _shape[0]; ++i, ++it_input) {
                    float d = dist(i, center[0]);
                    *it_input += A*exp(-d*d/(2.0 * sigma * sigma));
                }
            }
            ////////////////////////////
            //  2D
            else if(_shape.size() == 2) {
                auto dist = neuralfield::distances::make_euclidean_2D({this->_shape[0], this->_shape[1]}, _toric);
          
                auto it_input = _input.begin();
                for(int i = 0 ; i < _shape[0]; ++i) {
                    for(int j = 0 ; j < _shape[1]; ++j, ++it_input) {
                        double d = dist({double(i), double(j)}, {center[0], center[1]});
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

            for(int i = 0 ; i < _nb_gaussians; ++i) {

                for(unsigned int i = 0 ; i < _shape.size(); ++i)
                    center[i] = neuralfield::random::uniform(0, _shape[i]-1);

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


