#include "neuralfield.hpp"
#include "optimization-scenario.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

////////////
// Example of params with global inhibition
//
// 1D
// Non Toric
// ./examples/example-002b-optimize 0.05 0.01 0 50
//    ./examples/example-002-test 0.149815 3.58467 396.044 0.0283683 -138.285 0 15
//    ./examples/example-002-test 0.149815 3.58467 396.044 0.0283683 -138.285 0 30
//    ./examples/example-002-test 0.149815 3.58467 396.044 0.0283683 -138.285 0 50000

// Toric
// ./examples/example-002b-optimize 0.05 0.01 1 50
//
//   ./examples/example-002-test 0.198486 2.10177 311.587 0.0855248 -248.695 1 15
//   ./examples/example-002-test 0.198486 2.10177 311.587 0.0855248 -248.695 1 30
//   ./examples/example-002-test 0.198486 2.10177 311.587 0.0855248 -248.695 1 50000
//


////
// 2D
// Non Toric
//
// ./examples/example-002b-optimize 0.05 0.01 0 10 10
//    ./examples/example-002-test 0.175826 4.478 71207.3 0.0604805 -17505.6 0 10 10
//    ./examples/example-002-test 0.175826 4.478 71207.3 0.0604805 -17505.6 0 100 100

// Toric
//
// ./examples/example-002b-optimize 0.05 0.01 1 10 10
//
//    ./examples/example-002-test 0.190387 0.125372 24399 0.141612 -20897.5 1 10 10
//    ./examples/example-002-test 0.190387 0.125372 24399 0.141612 -20897.5 1 100 100

void fillInput(neuralfield::values_iterator begin,
        neuralfield::values_iterator end,
        const Input& x) {
    std::copy(x.begin(), x.end(), begin);
}

int main(int argc, char* argv[]) {

    if(argc != 8 && argc != 9) {
        std::cerr << "Usage : " << argv[0] << " dt_tau h Ap sp Am toric N <M>" << std::endl;
        return EXIT_FAILURE;
    }

    double dt_tau   = std::atof(argv[1]);
    double baseline = std::atof(argv[2]);
    double Ap       = std::atof(argv[3]);
    double sp       = std::atof(argv[4]);
    double Am       = std::atof(argv[5]);
    bool toric      = std::atoi(argv[6]);

    std::vector<int> shape;
    shape.push_back(std::atoi(argv[7]));
    if(argc == 9)
        shape.push_back(std::atoi(argv[8]));

    auto input = neuralfield::input::input<Input>(shape, fillInput, "input");
    auto h     = neuralfield::function::constant(baseline, shape, "h");
    auto u     = neuralfield::buffered::leaky_integrator(dt_tau, shape, "u");
    auto g_exc = neuralfield::link::gaussian(Ap, sp, toric, false, shape,"gexc");
    auto g_inh = neuralfield::link::full(Am, shape, "ginh");
    auto fu    = neuralfield::function::function("sigmoid", shape, "fu");
    auto noise = neuralfield::function::uniform_noise(-0.1, 0.1, shape, "noise");

    g_exc->connect(fu);
    g_inh->connect(fu);
    fu->connect(u);
    u->connect(g_exc + g_inh + input + h + noise);

    auto net = neuralfield::get_current_network();
    net->print();

    net->init();

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("f(u)" , cv::WINDOW_AUTOSIZE);

    std::cout << "Usage : " << std::endl;
    std::cout << "   <space> : pause/run " << std::endl;
    std::cout << "   <esc>   : quit " << std::endl;
    std::cout << "   <r>     : random input " << std::endl;
    std::cout << "   <s>     : structured input" << std::endl;
    std::cout << std::endl;

    bool stop = false;
    bool run = true;

    // We instanciante scenarios just for generating inputs
    // we do not care about the parameters of the fitness..
    auto s_random = RandomCompetition(0, shape, 0., 0., false);
    auto s_struct = StructuredCompetition(0, shape, 0., 0., false, 5., 1./5.);
    s_random.set_input(net);

    unsigned int width;
    unsigned int height;
    if(shape.size() == 2) {
        width = shape[0];
        height = shape[1];
    }
    else {
        width = 500;
        height = 500;
    }

    cv::Mat img_input;
    cv::Mat resized_input = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
    cv::Mat img_fu;
    cv::Mat resized_fu = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);

    int step = 0;
    while(!stop) {

        if(run) {
            net->step();
            ++step;
        }

        // Fill in the images
        if(shape.size() == 1) {
            cv::Point prev, next;
            
            {
                resized_input.setTo(255);
                auto it_input = net->get("input")->begin();
                auto it_end   = net->get("input")->end();
                int i = 0;
                prev.x = double(i)/shape[0] * width;
                prev.y = (1.-(*it_input))*height;
                ++it_input;
                ++i;
                while(it_input != it_end) {
                    next.x = double(i)/shape[0] * width;
                    next.y = (1.-(*it_input))*height;
                    cv::line(resized_input, prev, next, cv::Scalar(255, 0, 0));

                    prev = next;
                    ++it_input;
                    ++i;
                }
            }
 
            {
                resized_fu.setTo(255);
                auto it_fu  = net->get("fu")->begin();
                auto it_end = net->get("fu")->end();
                int i = 0;
                prev.x = double(i)/shape[0] * width;
                prev.y = (1.-(*it_fu))*height;
                ++it_fu;
                while(it_fu != it_end) { 
                    next.x = double(i)/shape[0] * width;
                    next.y = (1.-(*it_fu))*height;
                    cv::line(resized_fu, prev, next, cv::Scalar(255, 0, 0));
                    prev = next;
                    ++it_fu;
                    ++i;
                }
            }
        }
        else if(shape.size() == 2) {
            img_input = cv::Mat::zeros(height, width, CV_32F);
            img_fu = cv::Mat::zeros(height, width, CV_32F);

            auto it_input = net->get("input")->begin();
            for(unsigned int i = 0 ; i < height ; ++i)
                for(unsigned int j = 0 ; j < width; ++j, ++it_input)
                    img_input.at<float>(i,j) = *it_input;

            auto it_fu = net->get("fu")->begin();
            for(unsigned int i = 0 ; i < height ; ++i)
                for(unsigned int j = 0 ; j < width; ++j, ++it_fu)
                    img_fu.at<float>(i,j) = *it_fu;

            cv::resize(img_input, resized_input, cv::Size(500, 500));
            cv::resize(img_fu, resized_fu, cv::Size(500, 500));
        }
        std::string title = std::string("Step ") + std::to_string(step);
        std::cout << '\r' << title << std::flush;
        
        cv::imshow("Input", resized_input);
        cv::imshow("f(u)", resized_fu);

        // Handle the keyboard
        char key = (char)cv::waitKey(1);
        if(key == 27) { // 'ESC'
            std::cout << "<quit>" << std::endl;
            stop = true;
        }
        else if(key == 32) { // 'space'
            run = !run;
            if(run)
                std::cout << "<Running>" << std::endl;
            else
                std::cout << "<Paused>" << std::endl;

        }
        else if(key == 'r') {
            s_random.set_input(net);
            step = 0;
        }
        else if(key == 's') {
            s_struct.set_input(net);
            step = 0;
        }

    }

}
