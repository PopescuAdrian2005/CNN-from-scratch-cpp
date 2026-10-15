#pragma once
#include <cassert>
#include <cmath>
#include <random>

#include "cnn/Tensor.h"

namespace mat
{
    inline Matrix zeros(int rows, int cols)
    {
        return Matrix(rows,Vec(cols,0.0f));
    }
    inline Matrix random (int rows, int cols, int fanIn)
    {
        Matrix m = zeros(rows, cols);
        float norm =  std::sqrt(2.0f / fanIn);
        std::normal_distribution<float> dist(0.0f,norm);
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                m[i][j]=dist(gen);
            }
        }
        return m;
    }
    inline Vec multiply(const Matrix& w, const Vec& x)
    {
        assert(w.size() != 0 && w[0].size() == x.size());
        Vec y(w.size(),0.0f);
        for(size_t i = 0; i < y.size(); i++)
        {
            for(size_t j = 0; j < x.size(); j++)
            {
                y[i]+=w[i][j] * x[j];
            }
        }
        return y;
    }
    inline Vec add (const Vec&x ,const Vec& y)
    {
        assert(x.size() ==  y.size());
        Vec sum(x.size());
        for(size_t i = 0;i < x.size(); i++)
        {
            sum[i]=x[i]+y[i];
        }
        return sum;
    }
} 
