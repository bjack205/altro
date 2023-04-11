# Add libfmt
list(APPEND CMAKE_MESSAGE_CONTEXT fmt)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG a33701196adfad74917046096bf5a2aa0ab0bb50  # version 9.1.0
)
set(FMT_INSTALL ON CACHE BOOL "..." FORCE)
FetchContent_MakeAvailable(fmt)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

# Add Eigen
list(APPEND CMAKE_MESSAGE_CONTEXT eigen)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation." FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # version 3.4.0
)
FetchContent_MakeAvailable(eigen)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

# ArduinoEigen
if (ALTRO_ENABLE_ARDUINO_LIB_INSTALL)
  list(APPEND CMAKE_MESSAGE_CONTEXT arduinoeigen)
  FetchContent_Declare(arduinoeigen
    GIT_REPOSITORY https://github.com/hideakitai/ArduinoEigen.git
    GIT_TAG 0.2.3
  )
  FetchContent_MakeAvailable(arduinoeigen)
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endif()

# JSON
list(APPEND CMAKE_MESSAGE_CONTEXT json)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

## AltroCpp
#list(APPEND CMAKE_MESSAGE_CONTEXT altrocpp)
#FetchContent_Declare(altrocpp
#  GIT_REPOSITORY https://github.com/bjack205/altro-cpp.git
#  GIT_TAG edf246b0a1f9584cfb9ccf9f30afade6b039d2a7
#  )
#FetchContent_MakeAvailable(altrocpp)
#list(POP_BACK CMAKE_MESSAGE_CONTEXT)
