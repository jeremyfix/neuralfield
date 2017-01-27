# Generate the list of source files
# using GLOB does not RECURSE
file(
	GLOB
	source_files
	*.cpp
)

file(
	GLOB
	headers_files
	*.hpp
)

# Define our library 
add_library(
	neuralfield
	SHARED
	${source_files}
)
target_compile_options(neuralfield PUBLIC ${PROJECT_CFLAGS})
target_link_libraries(neuralfield  ${PROJECT_LIBS})


# Installation targets
#install (TARGETS neuralfield
#	 DESTINATION ${LIB_INSTALL_DIR}
#         COMPONENT binary)

install (FILES ${headers_files}
	 DESTINATION ${INCLUDE_INSTALL_DIR}
	 COMPONENT devel)
