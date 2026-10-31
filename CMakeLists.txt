cmake_minimum_required(VERSION 3.30)

project(TransformerCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(TransformerCPP main.cpp)