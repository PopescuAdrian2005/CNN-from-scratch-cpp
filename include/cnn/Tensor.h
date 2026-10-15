#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using Vec = std::vector<float>;
using Matrix = std::vector<Vec>;
inline std::mt19937 gen({std::random_device{}()});

struct ZeroInitTag {};

struct Tensor
{
    std::vector<float> vals;
    int channels = 0, rows = 0, cols = 0;
    Tensor() = default;
    Tensor(const Vec &newvals)
    {
        vals = newvals;
        channels = 1;
        rows = 1;
        cols = vals.size();
    }
    Tensor(int channels, int rows, int cols) 
    {
        this->channels = channels;
        this->rows = rows;
        this->cols = cols;
        vals = Vec(channels * rows * cols, 0.0f);
    }
    Tensor(const Tensor &prevtensor, int channels, int rows, int cols)
    {
        vals = prevtensor.vals;
        this->channels = channels;
        this->rows = rows;
        this->cols = cols;
    }
    Tensor(const Tensor &prevtensor, int padding)
    {
        channels = prevtensor.channels;
        rows = prevtensor.rows+2*padding;
        cols = prevtensor.cols+2*padding;
        vals=Vec(channels*rows*cols,0.0f);
        for (int in = 0; in < channels; in++)
            for (int i = 0; i < rows - 2*padding; i++)
                for (int j = 0; j < cols - 2*padding; j++)
                    vals[in*rows*cols+(i+padding)*cols+j+padding]=prevtensor.at(in, i, j);
    }
    Tensor(const Tensor &prevtensor, int padding,ZeroInitTag)
    {
        channels = prevtensor.channels;
        rows = prevtensor.rows+2*padding;
        cols = prevtensor.cols+2*padding;
        vals=Vec(channels*rows*cols,0.0f);
        for (int in = 0; in < channels; in++)
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    vals[in*rows*cols+i*cols+j]=prevtensor.at(in, i-padding, j-padding);
    }
    float at(int channel, int row, int col) const
    {
        return vals[channel * rows * cols + row * cols + col];
    }
    float &at(int channel, int row, int col)
    {
        return vals[channel * rows * cols + row * cols + col];
    }
    float at(int poz) const
    {
        return vals[poz];
    }
    void gradient(int label)
    {
        vals[label]-=1;
    }
};

struct Tensor4D
{
    std::vector<float> vals;
    int filtersin = 0, rows = 0, cols = 0, filtersout = 0;
    Tensor4D() = default;
    Tensor4D(int filterout, int filterin, int row, int col)
    {
        this->filtersin = filterin;
        this->filtersout = filterout;
        this->rows = row;
        this->cols = col;
        vals = Vec(filtersin * filtersout * rows * cols);
        int fanIn = row * col * filterin;
        float norm = std::sqrt(2.0f / fanIn);
        std::normal_distribution<float> dist(0.0f, norm);
        for (int i = 0; i < filtersin * filtersout * rows * cols; i++)
        {
            vals[i] = dist(gen);
        }
    }
    Tensor4D(int filtersout, int filtersin, int rows, int cols, ZeroInitTag)
    {
        this->filtersin = filtersin;
        this->filtersout = filtersout;
        this->rows = rows;
        this->cols = cols;
        vals = Vec(filtersin * filtersout * rows * cols, 0.0f);
    }
    float at(int filterout, int filterin, int row, int col) const
    {
        return vals[filterout * filtersin * rows * cols + filterin * rows * cols + row * cols + col];
    }
    float &at(int filterout, int filterin, int row, int col)
    {
        return vals[filterout * filtersin * rows * cols + filterin * rows * cols + row * cols + col];
    }
    float at(int poz)
    {
        return vals[poz];
    }
};