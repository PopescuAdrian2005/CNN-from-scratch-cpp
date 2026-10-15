#pragma once
#include <algorithm>
#include <cassert>
#include <memory>

#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"

struct ReluLayer :ILayer
{
    Tensor activation;
    ReluLayer() = default;
    std::unique_ptr<ILayer> clone() override
    {
        return std::make_unique<ReluLayer>(*this);
    }
    Tensor forward(const Tensor &input) override
    {
        activation = input;
        for (size_t i = 0; i < activation.vals.size(); i++)
        {
            activation.vals[i] = std::max(activation.vals[i], 0.0f);
        }
        return activation;
    }
    Tensor backward(const Tensor &nextlayer) override
    {
        assert(activation.channels == nextlayer.channels);
        assert(activation.rows == nextlayer.rows);
        assert(activation.cols == nextlayer.cols);
        Tensor currlayer=nextlayer;
        for(size_t i=0;i<currlayer.vals.size();i++)
        {
            if(activation.vals[i]<=0.0f)
            {
                currlayer.vals[i]=0;
            }
            else
            {
                currlayer.vals[i]=nextlayer.vals[i];
            }
        }
        return currlayer;
    }
};