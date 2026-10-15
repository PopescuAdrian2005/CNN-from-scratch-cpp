#pragma once
#include <cassert>
#include <memory>

#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"

struct FlattenLayer :ILayer
{
    int channels=0,rows=0,cols=0;
    FlattenLayer() = default;
    Tensor forward(const Tensor &input) override
    {
        channels=input.channels;
        rows=input.rows;
        cols=input.cols;
        return Tensor(input.vals);
    }
    Tensor backward(const Tensor &gradient) override
    {
        assert((int)gradient.vals.size() == (channels * rows * cols));
        return Tensor(gradient,channels,rows,cols);
    }
    std::unique_ptr<ILayer> clone() override
    {
        return std::make_unique<FlattenLayer>(*this);
    }
};