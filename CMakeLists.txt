cmake_minimum_required(VERSION 3.19)

set (CMAKE_CXX_STANDARD 20)

project(pathtracer)

find_package(CUDAToolkit)

set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.4 COMPONENTS system window graphics audio main REQUIRED)

add_executable(pathtracer "src/main.cpp")

target_include_directories(pathtracer
    PUBLIC include
    PUBLIC jitify
)

target_link_libraries(pathtracer
    PUBLIC CUDA::toolkit
    PUBLIC CUDA::cudart
    PUBLIC CUDA::nvrtc
	PUBLIC sfml-system
	PUBLIC sfml-window
	PUBLIC sfml-graphics
	PUBLIC sfml-audio
	PUBLIC sfml-main
)

# TODO: this hard-codes full paths into the executable (I think). Is this needed?
target_compile_definitions(pathtracer PUBLIC "-DCUDA_INC_DIR=\"${CUDAToolkit_INCLUDE_DIRS}\"")

# Copy data directory to build directory
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "$<TARGET_FILE_DIR:pathtracer>/data"
)

if(MSVC)
    # Needed to prevent linker error about UnDecorateSymbolName from <dbghelp.h>
    target_link_libraries(pathtracer
        PUBLIC Dbghelp 
    )

    # Start the executable in the build directory
    set_target_properties(
        pathtracer PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:pathtracer>"
    )

	# increase warning level
	add_definitions(/W4)

	# turn warning C4715 ("not all control paths return a value") into an error
	add_definitions(/we4715)

	# disable warning C4250 ("class X inherits virtual member function Y via dominance")
	add_definitions(/wd4250)

	# turn warning C4239 (non-standard extension that allows temporaries to be bound to
	# non-const references, yay microsoft) into an error
	# This extension is without question the source of many bugs and disastrous surprises
	add_definitions(/we4239)

    # Conformance mode (to heck with all silly Microsoft language extensions)
    add_definitions(/permissive-)
else()
	SET(CMAKE_CXX_FLAGS -pthread)
endif()