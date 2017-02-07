# Make sure the compiler can find include files from our library.
include_directories (${CMAKE_SOURCE_DIR}/src
  ${POPOT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

# Define our examples to compile
file(
	GLOB 
	usage_examples
	*.cpp
)
MESSAGE("Opencv : ${OpenCV_INCLUDE_DIRS}")

# loop over the list to compile them
foreach(f ${usage_examples})
    get_filename_component(exampleName ${f} NAME_WE) 
    add_executable (${exampleName} ${f}) 
    target_compile_options(${exampleName} PUBLIC ${PROJECT_CFLAGS})
    target_link_libraries(${exampleName} ${PROJECT_LIBS} neuralfield ${OpenCV_LIBS})

    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${exampleName} 
            DESTINATION bin
            RENAME ${CMAKE_PROJECT_NAME}-${exampleName}
	    COMPONENT binary)
endforeach(f)
