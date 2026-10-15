#pragma once
#include <cassert>
#include <fstream>
#include <memory>

#include "cnn/Matrix.h"
#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"

struct ConvWeights
{
    Tensor4D weights;
    Vec bias;
};

struct ConvLayer : ILayer
{
    int outputChannels, inputChannels, rows, cols, padding;
    ConvWeights parameters;
    ConvWeights gradients;
    Tensor activation;
    ConvLayer() = default;
    ConvLayer(int filterout, int filterin, int row, int col, int padd) : outputChannels(filterout), inputChannels(filterin), rows(row), cols(col), padding(padd)
    {
        assert(padd >= 0);
        parameters.weights = Tensor4D(outputChannels, inputChannels, rows, cols);
        parameters.bias = Vec(outputChannels, 0.0f);
        gradients.weights = Tensor4D(outputChannels, inputChannels, rows, cols, ZeroInitTag{});
        gradients.bias = Vec(outputChannels, 0.0f);
    }
    ConvLayer(int filterout, int filterin, int row, int col, int padd, ZeroInitTag) : outputChannels(filterout), inputChannels(filterin), rows(row), cols(col), padding(padd)
    {
        assert(padd >= 0);
        parameters.weights = Tensor4D(outputChannels, inputChannels, rows, cols, ZeroInitTag{});
        parameters.bias = Vec(outputChannels, 0.0f);
        gradients.weights = Tensor4D(outputChannels, inputChannels, rows, cols, ZeroInitTag{});
        gradients.bias = Vec(outputChannels, 0.0f);
    }
    ConvLayer(std::ifstream &fd, int filterout, int filterin, int row, int col, int padd) : ConvLayer(filterout, filterin, row, col, padd)
    {
        load(fd);
    }
    std::unique_ptr<ILayer> clone() override
    {
        return std::make_unique<ConvLayer>(*this);
    }
    int size() override
    {
        return outputChannels * inputChannels * rows * cols + outputChannels;
    }
    void load(std::ifstream &fd) override
    {
        for (int out = 0; out < outputChannels; out++)
        {
            for (int in = 0; in < inputChannels; in++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        float x;
                        if (!fd.read(reinterpret_cast<char *>(&x), sizeof(float)))
                            return;
                        parameters.weights.at(out, in, i, j) = x;
                    }
                }
            }
        }
        for (int i = 0; i < outputChannels; i++)
        {
            float x;
            if (!fd.read(reinterpret_cast<char *>(&x), sizeof(float)))
                return;
            parameters.bias[i] = x;
        }
    }
    void save(std::ofstream &fd) override
    {
        for (int out = 0; out < outputChannels; out++)
        {
            for (int in = 0; in < inputChannels; in++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        fd.write(reinterpret_cast<const char *>(&parameters.weights.at(out, in, i, j)), sizeof(float));
                    }
                }
            }
        }
        for (int i = 0; i < outputChannels; i++)
        {
            fd.write(reinterpret_cast<const char *>(&parameters.bias[i]), sizeof(float));
        }
    }
    void update(float learningRate, int batchSize, float weightDecay) override
    {
        assert(batchSize != 0);
        learningRate = learningRate / batchSize;
        weightDecay = weightDecay * batchSize;
        for (int out = 0; out < outputChannels; out++)
        {
            for (int in = 0; in < inputChannels; in++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        parameters.weights.at(out, in, i, j) -= (weightDecay * parameters.weights.at(out, in, i, j) + gradients.weights.at(out, in, i, j)) * learningRate;
                        gradients.weights.at(out, in, i, j) = 0;
                    }
                }
            }
        }
        for (int i = 0; i < outputChannels; i++)
        {
            parameters.bias[i] -= gradients.bias[i] * learningRate;
            gradients.bias[i] = 0;
        }
    }
    Tensor forward(const Tensor &input) override
    {
        assert(input.channels == inputChannels && input.rows + 2 * padding >= rows && input.cols + 2 * padding >= cols);
        activation = Tensor(input, padding);
        Tensor output(outputChannels, activation.rows - rows + 1, activation.cols - cols + 1);
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int out = 0; out < outputChannels; out++)
            {
                for (int i = 0; i < output.rows; i++)
                {
                    for (int j = 0; j < output.cols; j++)
                    {
                        float sum = parameters.bias[out];
                        for (int in = 0; in < inputChannels; in++)
                        {
                            for (int dx = 0; dx < rows; dx++)
                            {
                                for (int dy = 0; dy < cols; dy++)
                                {
                                    sum += activation.at(in, i + dx, j + dy) * parameters.weights.at(out, in, dx, dy);
                                }
                            }
                        }
                        output.at(out, i, j) = sum;
                    }
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor &nextlayer) override
    {
        assert(outputChannels == nextlayer.channels);
        Tensor currlayer(inputChannels, activation.rows, activation.cols);
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int in = 0; in < inputChannels; in++)
            {
                for (int out = 0; out < outputChannels; out++)
                {
                    for (int dx = 0; dx < rows; dx++)
                    {
                        for (int dy = 0; dy < cols; dy++)
                        {
                            float val = parameters.weights.at(out, in, dx, dy);
                            for (int i = 0; i < nextlayer.rows; i++)
                            {
                                int basecurr = currlayer.cols * (currlayer.rows * in + i + dx) + dy;
                                int basenext = nextlayer.cols * (nextlayer.rows * out + i);
                                for (int j = 0; j < nextlayer.cols; j++)
                                {
                                    currlayer.vals[basecurr + j] += nextlayer.vals[basenext + j] * val;
                                }
                            }
                        }
                    }
                }
            }
#pragma omp for schedule(static)
            for (int out = 0; out < outputChannels; out++)
            {
                for (int in = 0; in < inputChannels; in++)
                {
                    for (int dx = 0; dx < rows; dx++)
                    {
                        for (int dy = 0; dy < cols; dy++)
                        {
                            float sum = 0;
                            for (int i = 0; i < nextlayer.rows; i++)
                            {
                                int base = (out * nextlayer.rows + i) * nextlayer.cols;
                                for (int j = 0; j < nextlayer.cols; j++)
                                {
                                    sum += activation.at(in, i + dx, j + dy) * nextlayer.at(base + j);
                                }
                            }
                            gradients.weights.at(out, in, dx, dy) += sum;
                        }
                    }
                }
            }
#pragma omp for schedule(static)
            for (int out = 0; out < outputChannels; out++)
            {
                for (int i = 0; i < nextlayer.rows; i++)
                {
                    for (int j = 0; j < nextlayer.cols; j++)
                    {
                        gradients.bias[out] += nextlayer.at(out, i, j);
                    }
                }
            }
        }
        Tensor output(currlayer, -padding, ZeroInitTag{});
        return output;
    }
};