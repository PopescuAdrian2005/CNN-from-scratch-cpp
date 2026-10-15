#pragma once
#include <cassert>
#include <fstream>
#include <memory>

#include "cnn/Matrix.h"
#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"

struct DenseWeights
{
    Matrix weights;
    Vec bias;
};

struct DenseLayer :ILayer
{
    int insize, outsize;
    DenseWeights parameters;
    DenseWeights gradients;
    Vec activation;
    DenseLayer() = default;
    DenseLayer(int insize, int outsize)
    {
        this->insize = insize;
        this->outsize = outsize;
        parameters.weights = mat::random(outsize, insize, insize);
        parameters.bias = Vec(outsize, 0.0f);
        gradients.weights = mat::zeros(outsize, insize);
        gradients.bias = Vec(outsize, 0.0f);
    }
    DenseLayer(int insize, int outsize, ZeroInitTag)
    {
        this->insize = insize;
        this->outsize = outsize;
        parameters.weights = mat::zeros(outsize, insize);
        parameters.bias = Vec(outsize, 0.0f);
        gradients.weights = mat::zeros(outsize, insize);
        gradients.bias = Vec(outsize, 0.0f);
    }
    DenseLayer(std::ifstream &fd,int insize,int outsize) :DenseLayer(insize,outsize)
    {
        load(fd);
    }
    std::unique_ptr<ILayer> clone() override
    {
        return std::make_unique<DenseLayer>(*this);
    }
    int size() 
    {
        return insize*outsize+outsize;
    }
    void load(std::ifstream &fd) override    
    {
        this->insize=insize;
        this->outsize=outsize;
        parameters.weights = mat::zeros(outsize, insize);
        parameters.bias = Vec(outsize, 0.0f);
        gradients.weights = mat::zeros(outsize, insize);
        gradients.bias = Vec(outsize, 0.0f);
        for (int i = 0; i < outsize; i++)
        {
            for (int j = 0; j < insize; j++)
            {
                float x;
                fd.read(reinterpret_cast<char *>(&x), sizeof(float));
                parameters.weights[i][j] = x;
            }
        }
        for (int i = 0; i < outsize; i++)
        {
            float x;
            fd.read(reinterpret_cast<char *>(&x), sizeof(float));
            parameters.bias[i] = x;
        }
    }
    void save(std::ofstream &fd) override
    {
        for (int i = 0; i < outsize; i++)
        {
            for (int j = 0; j < insize; j++)
            {
                if(!fd.write(reinterpret_cast<const char *>(&parameters.weights[i][j]), sizeof(float)))
                    return;
            }
        }
        for (int i = 0; i < outsize; i++)
        {
            if(!fd.write(reinterpret_cast<const char *>(&parameters.bias[i]), sizeof(float)))
                return; 
        }
    }
    void update(float learningRate, int batchSize,float weightdecay) override
    {
        assert(batchSize!=0);
        learningRate = learningRate / batchSize;
        weightdecay=weightdecay*batchSize;
        for (int i = 0; i < outsize; i++)
        {
            for (int j = 0; j < insize; j++)
            {
                parameters.weights[i][j] -= (weightdecay*parameters.weights[i][j]+gradients.weights[i][j]) * learningRate;
                gradients.weights[i][j] = 0;
            }
        }
        for (int i = 0; i < outsize; i++)
        {
            parameters.bias[i] -= gradients.bias[i] * learningRate;
            gradients.bias[i] = 0;
        }
    }
    Tensor forward(const Tensor &input) override
    {
        assert(insize == input.cols);
        activation=input.vals;
        return Tensor(mat::add(mat::multiply(parameters.weights, activation),parameters.bias));
    }
    Tensor backward(const Tensor &nextlayer) override
    {
        assert(outsize == nextlayer.cols);
        assert((int)activation.size() == insize);
        Vec currlayer(insize, 0.0f);
        for (int i = 0; i < insize; i++)
        {
            for (int j = 0; j < outsize; j++)
            {
                currlayer[i] += nextlayer.at(j) * parameters.weights[j][i];
                gradients.weights[j][i] += nextlayer.at(j)*activation[i];
            }
        }
        for(int j = 0; j < outsize ;j++)
        {
            gradients.bias[j] += nextlayer.at(j);
        }
        return Tensor(currlayer);
    }
};