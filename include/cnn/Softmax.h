#pragma once
#include <algorithm>
#include <cmath>

#include "cnn/Tensor.h"

inline Tensor softmax(const Tensor &x)
{
    Tensor out(1,1,x.cols);
    float maxV = *std::max_element(x.vals.begin(), x.vals.end());
    float sum = 0.0f;
    for (size_t i = 0; i < x.vals.size(); i++)
    {
        out.vals[i] = std::exp(x.vals[i] - maxV);
        sum += out.vals[i];
    }
    for (size_t i = 0; i < x.vals.size(); i++)
    {
        out.vals[i] = out.vals[i] / sum;
    }
    return out;
}
