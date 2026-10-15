#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "cnn/Softmax.h"
#include "cnn/Tensor.h"
#include "cnn/layers/ILayer.h"
#include "cnn/training/DataAugmentation.h"
#include "cnn/data/Dataset.h"
struct Cnn
{
    std::vector<std::unique_ptr<ILayer>> layers;
    Cnn() = default;
    void load(std::string &path)
    {
        std::ifstream fd(path, std::ios::binary | std::ios::ate);
        if (!fd.is_open())
            return;
        auto filesz = fd.tellg();
        int netsize = 0;
        for (auto &layer : layers)
        {
            netsize += layer->size();
        }
        std::streamoff expectedBytes = sizeof(float) * (netsize);
        if (filesz != expectedBytes)
            return;
        fd.seekg(0, std::ios::beg);

        std::vector<std::unique_ptr<ILayer>> tmp;
        for (auto &layer : layers)
        {
            tmp.push_back(layer->clone());
            tmp.back()->load(fd);
        }
        if (!fd)
        {
            return;
        }
        for (int i = 0; i < (int)layers.size(); i++)
        {
            auto &layer = layers[i];
            layer = std::move(tmp[i]);
        }
    }
    Tensor forward(Tensor &input)
    {
        Tensor x = input;
        for (size_t i = 0; i < layers.size(); i++)
        {
            x = layers[i]->forward(x);
        }
        return softmax(x);
    }
    void backward(Tensor &input, int label)
    {
        Tensor g = forward(input);
        g.gradient(label);
        for (int i = (int)layers.size() - 1; i >= 0; i--)
        {
            g = layers[i]->backward(g);
        }
    }
    void update(float learningRate, int batchSize, float weightDecay)
    {
        for (auto &layer : layers)
        {
            layer->update(learningRate, batchSize, weightDecay);
        }
    }
    void train(Dataset &train, int epochs, int batchSize, float learningRate, float learningRateFactor, float weightDecay, bool useDataAugmentation, int epochsBeforeSave, std::string &path,AugmentationParameters &dataAgumentRates)
    {
        if (epochs < 0 || batchSize <= 0)
        {
            std::cerr << "Invalid training parameters.\n";
            return;
        }
        Dataset epochDataset;
        for (int i = 0; i < epochs; i++)
        {
            std::cout << "Epoch: " << i + 1 << '/' << epochs << '\n';
            if (useDataAugmentation == true)
            {
                epochDataset.images = ModifyData(train.images,dataAgumentRates);
                epochDataset.labels = train.labels;
                shuffleDataset(epochDataset.images, epochDataset.labels);
            }
            else
            {
                epochDataset = train;
            }
            trainEpoch(epochDataset.images, epochDataset.labels, batchSize, learningRate, weightDecay);
            learningRate = learningRate * learningRateFactor;
            if (epochsBeforeSave > 0 && (i + 1) % epochsBeforeSave == 0)
            {
                save(path);
            }
        }
    }
    void trainEpoch(std::vector<Tensor> &input, std::vector<int> &labels, int batchSize, float learningRate, float weightDecay)
    {
        int acc = 0;
        int batchid = 0;
        for (size_t i = 0; i < input.size(); i++)
        {
            acc++;
            backward(input[i], labels[i]);
            if ((i + 1) % batchSize == 0 || i + 1 == input.size())
            {
                batchid++;
                update(learningRate, acc, weightDecay);
                acc = 0;
            }
        }
    }
    void save(std::string &path)
    {
        std::ofstream fd(path, std::ios::binary | std::ios::trunc);
        for (auto &layer : layers)
        {
            layer->save(fd);
        }
    }
    float crossEntropy(Tensor &input, int label)
    {
        float value = std::max((float)1e-7, forward(input).at(label));
        return -log(value);
    }
    float accuracy(Tensor &input, int label)
    {
        Tensor output = forward(input);
        for (int i = 0; i < output.cols; i++)
        {
            if ((output.vals[i] >= output.vals[label] && i != label) || (!std::isfinite(output.vals[i])))
            {
                return 0;
            }
        }
        return 1;
    }
    float averageCrossEntropy(std::vector<Tensor> &input, std::vector<int> &labels)
    {
        assert(input.size() > 0);
        float sum = 0;
        for (size_t i = 0; i < input.size(); i++)
        {
            sum += crossEntropy(input[i], labels[i]);
        }
        return sum / input.size();
    }
    float averageAccuracy(std::vector<Tensor> &input, std::vector<int> &labels)
    {
        assert(input.size() > 0);
        float sum = 0;
        for (size_t i = 0; i < input.size(); i++)
        {
            sum += accuracy(input[i], labels[i]);
        }
        return sum * 100 / input.size();
    }
};
