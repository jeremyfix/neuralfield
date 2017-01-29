
CMAKE_MINIMUM_REQUIRED( VERSION 2.8.4 )

project (neuralfield) 

SET(PACKAGE_VERSION "1.00")
SET(PACKAGE_LICENSE "GPL")

SET(CMAKE_BUILD_TYPE Debug)

SET(PROJECT_DESCRIPTION_SUMMARY "C++ Library for parametrized NeuralFields")

#######################################
# Get the libs/inc for the dependencies
#######################################

find_package(PkgConfig)

pkg_check_modules(FFTW fftw3 REQUIRED)
pkg_check_modules(POPOT popot REQUIRED)

set(PKG_CONFIG_DEPENDS "fftw3 popot")

SET(PROJECT_LIBS "${FFTW_LDFLAGS} ${POPOT_LDFLAGS}")
SET(PROJECT_CFLAGS -Wall -std=c++11)

###################################
# Some prefix definitions
###################################

# Define the prefix where libs are going to be installed
SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Library directory name")

# Define the path to install the headers
SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/" CACHE PATH "The subdirectory to the header prefix")

# Define the path to install the documentation
SET(DOC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/doc/${CMAKE_PROJECT_NAME}" CACHE PATH "The subdirectory to the documentation prefix")

# Define the path to install the pc pkg-config file
SET(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/pkgconfig/" CACHE STRING "Base directory for pkgconfig files")

###################################
# Pkg-config stuff
###################################

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
"
Name: ${CMAKE_PROJECT_NAME}
Description: ${PROJECT_DESCRIPTION_SUMMARY}
Version: ${PACKAGE_VERSION}
Requires: ${PKG_CONFIG_DEPENDS}
Libs: -L${LIB_INSTALL_DIR} -lneuralfield
Cflags: -I${INCLUDE_INSTALL_DIR} ${PROJECT_CFLAGS}
"
)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
    	DESTINATION ${PKGCONFIG_INSTALL_DIR}
	COMPONENT devel)

###################################
#  Subdirectories
###################################
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(doc)


###################################
#  Custom command for source tgz
###################################

SET(DIST_DIR "${CMAKE_PROJECT_NAME}-${PACKAGE_VERSION}")
ADD_CUSTOM_TARGET(dist 
		  COMMAND rm -rf ${DIST_DIR}
		  COMMAND mkdir  ${DIST_DIR}
		  COMMAND cp -r ${CMAKE_SOURCE_DIR}/* ${DIST_DIR} || true 
		  COMMAND rm -rf ${DIST_DIR}/build
		  COMMAND mkdir ${DIST_DIR}/build
  		  COMMAND tar --exclude="*~" --exclude="._*" --exclude=".svn" -zcvf ${DIST_DIR}.tar.gz ${DIST_DIR}
		  COMMAND rm -rf  ${DIST_DIR}
    	          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

