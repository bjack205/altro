add_subdirectory(altrocpp_interface)
add_subdirectory(altro)

## TODO: Use target_sources(FILE_SET)  (new in 3.23)
#set(ALTRO_PUBLIC_HEADERS
#  altro/altro.hpp
#  altro/altro_solver.hpp
#  altro/solver/solver_options.hpp
#  altro/solver/exceptions.hpp
#  altro/solver/typedefs.hpp
#)
#cmake_print_variables(ALTRO_PUBLIC_HEADERS)
#install(FILES ${ALTRO_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
