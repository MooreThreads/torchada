#pragma once

#include <musa.h>

#ifndef CUresult
#define CUresult MUresult
#endif
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS MUSA_SUCCESS
#endif
#ifndef cuGetErrorString
#define cuGetErrorString muGetErrorString
#endif
