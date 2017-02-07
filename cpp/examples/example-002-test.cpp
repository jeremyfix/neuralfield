#include "neuralfield.hpp"
#include "optimization-scenario.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>

// Example of params :
// 1D
//  NonToric / NonScaled , 50
//   ./examples/example-002-test 0.161126 1.61908 1.87441 2.35995 -0.970397 73.4035 0 0 50

// 2D
//  Toric / Unscaled 30x30
//   ./examples/example-002-test 0.590007 1.65246 1.26578 7.71841 -1.14564 21.6151 1 0 30 30

void fillInput(neuralfield::values_iterator begin,
	       neuralfield::values_iterator end,
	       const Input& x) {
  std::copy(x.begin(), x.end(), begin);
}

int main(int argc, char* argv[]) {

  if(argc != 10 && argc != 11) {
    std::cerr << "Usage : " << argv[0] << " dt_tau h Ap sp Am sm toric scale N <M>" << std::endl;
    return EXIT_FAILURE;
  }

  double dt_tau = std::atof(argv[1]);
  double baseline = std::atof(argv[2]);
  double Ap = std::atof(argv[3]);
  double sp = std::atof(argv[4]);
  double Am = std::atof(argv[5]);
  double sm = std::atof(argv[6]);

  bool toric = std::atoi(argv[7]);
  bool scale = std::atoi(argv[8]);

  std::vector<int> shape;

  shape.push_back(std::atoi(argv[9]));
  if(argc == 11)
    shape.push_back(std::atoi(argv[10]));

  auto input = neuralfield::input::input<Input>(shape, fillInput, "input");

  auto h = neuralfield::function::constant(baseline, shape, "h");
  auto u = neuralfield::buffered::leaky_integrator(dt_tau, shape, "u");
  auto g_exc = neuralfield::link::gaussian(Ap, sp, toric, scale, shape,"gexc");
  auto g_inh =  neuralfield::link::gaussian(Am, sm, toric, scale, shape, "ginh");
  auto fu = neuralfield::function::function("sigmoid", shape, "fu");

  g_exc->connect(fu);
  g_inh->connect(fu);
  fu->connect(u);
  u->connect(g_exc + g_inh + input + h);
  
  auto net = neuralfield::get_current_network();
  net->print();
  
  net->init();

  cv::namedWindow("Input", cv::WINDOW_NORMAL);
  cv::namedWindow("f(u)", cv::WINDOW_NORMAL);

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
  auto s_struct = StructuredCompetition(0, shape, 0., 0., false, 5., shape[0]/5.);

  unsigned int width = shape[0];
  unsigned int height;
  if(shape.size() == 2)
    height = shape[1];
  else
    height = 3./4. * width;
    
  cv::Mat img_input;
  cv::Mat img_fu;

  int step = 0;
  while(!stop) {

    if(run) {
      //if(step < 100) {
      net->step();
      ++step;
	//}
    }

    // Fill in the images
    if(shape.size() == 1) {
      cv::Point prev, next;
      
      img_input = cv::Mat::zeros(height, width, CV_8UC3);
      img_input.setTo(255);
      auto it_input = net->get("input")->begin();
      next.x = 0;
      next.y = (1.-(*it_input))*height;
      ++it_input;
      for(unsigned int i = 1 ; i < width; ++i, ++it_input) {
	prev = next;
	next.x = i;
	next.y = (1.-(*it_input))*height;
	cv::line(img_input, prev, next, cv::Scalar(255, 0, 0));
      }

      img_fu = cv::Mat::zeros(height, width, CV_8UC3);
      img_fu.setTo(255);
      auto it_fu = net->get("fu")->begin();
      next.x = 0;
      next.y = (1.-(*it_fu))*height;
      ++it_input;
      for(unsigned int i = 1 ; i < width; ++i, ++it_fu) {
	prev = next;
	next.x = i;
	next.y = (1.-(*it_fu))*height;
	cv::line(img_fu, prev, next, cv::Scalar(255, 0, 0));
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
      
    }
    std::string title = std::string("Step ") + std::to_string(step);
    std::cout << '\r' << title << std::flush;
    
    cv::imshow("Input", img_input);
    cv::imshow("f(u)", img_fu);
    
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
