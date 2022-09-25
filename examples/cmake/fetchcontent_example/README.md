# FetchContent Example

To run, follow these steps, starting from this directory:
```shell
mkdir build
cd build
cmake -G Ninja --log-context ..
cmake --build . -j4
```
Note that `-G Ninja` and `--log-context` are not required flags, but simply the author's 
preference for the Ninja generator and enabling better print statement in the CMake output.