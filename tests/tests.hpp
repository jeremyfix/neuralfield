#pragma once

#include <limits>
#include <stdexcept>
#include <string>


template<typename LAYER1, typename LAYER2>
inline void test_equal_layers(const LAYER1& l1, const LAYER2& l2, const std::string& file, int line, const std::string& pretty_function) {
	auto it_l2 = l2.cbegin();
    unsigned int idx = 0;
	for(const auto& v1: l1) {
		if(fabs(v1 - *it_l2) > std::numeric_limits<double>::epsilon()) 
			throw std::runtime_error(file
					+ std::string(":")
					+ std::to_string(line)
				    + std::string(" in ") 
					+ pretty_function 
					+ std::to_string(v1) 
					+ std::string("!=") 
                    + std::to_string(*it_l2) 
                    + std::string(" at index=")
                    + std::to_string(idx));
		++it_l2;
        ++idx;
	}
}

#define ASSERT_EQUAL_LAYERS( x, y ) \
{						   \
	test_equal_layers(x, y, std::string(__FILE__), __LINE__, std::string( __PRETTY_FUNCTION__ )); \
}

