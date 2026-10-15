#pragma once
#include <cassert>
#include <memory>

#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"

inline int dxstride2[4] = {0, 0, 1, 1};
inline int dystride2[4] = {0, 1, 0, 1};

struct MaxPoolLayer : ILayer
{
    int stride = 2;
    Tensor activation;
    MaxPoolLayer() = default;
    std::unique_ptr<ILayer> clone() override
    {
        return std::make_unique<MaxPoolLayer>(*this);
    }
    Tensor forward(const Tensor &input) override
    {
        activation = Tensor(input.channels, input.rows, input.cols);
        assert(input.rows % 2 == 0 && input.cols % 2 == 0);
        Tensor output(input.channels, input.rows / 2, input.cols / 2);
        for (int c = 0; c < output.channels; c++)
        {
            for (int i = 0; i < output.rows; i++)
            {
                for (int j = 0; j < output.cols; j++)
                {
                    int x = i * 2;
                    int y = j * 2;
                    int index = 0;
                    float best = input.at(c, x, y);
                    for (int z = 1; z < 4; z++)
                    {
                        int nx = x + dxstride2[z];
                        int ny = y + dystride2[z];
                        if (input.at(c, nx, ny) > best)
                        {
                            best = input.at(c, nx, ny);
                            index = z;
                        }
                    }
                    activation.at(c, x + dxstride2[index], y + dystride2[index]) = 1;
                    output.at(c, i, j) = best;
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor &nextlayer) override
    {
        assert(activation.channels == nextlayer.channels);
        assert(activation.rows == 2 * nextlayer.rows);
        assert(activation.cols == 2 * nextlayer.cols);
        Tensor currlayer = activation;
        for (int c = 0; c < currlayer.channels; c++)
        {
            for (int i = 0; i < currlayer.rows; i++)
            {
                for (int j = 0; j < currlayer.cols; j++)
                {
                    currlayer.at(c,i,j)=nextlayer.at(c,i/2,j/2)*activation.at(c,i,j);
                }
            }
        }
        return currlayer;
    }
};