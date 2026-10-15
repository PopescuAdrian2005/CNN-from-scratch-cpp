#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "cnn/Cnn.h"
#include "cnn/data/Cifar10Loader.h"
#include "cnn/data/MnistLoader.h"
#include "cnn/layers/ConvLayer.h"
#include "cnn/layers/DenseLayer.h"
#include "cnn/layers/FlattenLayer.h"
#include "cnn/layers/MaxPoolLayer.h"
#include "cnn/layers/ReluLayer.h"

Cnn createCifar10Model()
{
    Cnn net;

    net.layers.push_back(std::make_unique<ConvLayer>(16, 3, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<ConvLayer>(16, 16, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<MaxPoolLayer>());

    net.layers.push_back(std::make_unique<ConvLayer>(32, 16, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<ConvLayer>(32, 32, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<MaxPoolLayer>());

    net.layers.push_back(std::make_unique<ConvLayer>(64, 32, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<ConvLayer>(64, 64, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<MaxPoolLayer>());

    net.layers.push_back(std::make_unique<FlattenLayer>());
    net.layers.push_back(std::make_unique<DenseLayer>(4 * 4 * 64, 128));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<DenseLayer>(128, 10));

    return net;
}

Cnn createMnistModel()
{
    Cnn net;

    net.layers.push_back(std::make_unique<ConvLayer>(8, 1, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<ConvLayer>(8, 8, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<MaxPoolLayer>());

    net.layers.push_back(std::make_unique<ConvLayer>(16, 8, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<ConvLayer>(16, 16, 3, 3, 1));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<MaxPoolLayer>());

    net.layers.push_back(std::make_unique<FlattenLayer>());
    net.layers.push_back(std::make_unique<DenseLayer>(7 * 7 * 16, 128));
    net.layers.push_back(std::make_unique<ReluLayer>());
    net.layers.push_back(std::make_unique<DenseLayer>(128, 10));

    return net;
}

bool validWeightsFile(const std::string &path, Cnn &net)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Could not open weights file: " << path << '\n';
        return false;
    }

    std::streamoff expectedBytes = 0;
    for (const auto &layer : net.layers)
    {
        expectedBytes += static_cast<std::streamoff>(layer->size()) * sizeof(float);
    }

    if (file.tellg() != expectedBytes)
    {
        std::cerr << "The weights file is not compatible with the selected model.\n";
        return false;
    }

    file.seekg(0, std::ios::beg);
    float value = 0.0f;
    while (file.read(reinterpret_cast<char *>(&value), sizeof(value)))
    {
        if (!std::isfinite(value))
        {
            std::cerr << "The weights file contains NaN or infinity: "
                      << path << '\n';
            return false;
        }
    }

    return file.eof();
}

void printUsage(const char *programName)
{
    std::cerr
        << "Usage:\n"
        << "  " << programName << " <dataset> <weights> <fresh|resume>\n\n"
        << "Available datasets: mnist, cifar10\n\n"
        << "Examples:\n"
        << "  " << programName << " cifar10 weights/cifar10.bin fresh\n"
        << "  " << programName << " cifar10 weights/cifar10.bin resume\n"
        << "  " << programName << " mnist weights/mnist.bin fresh\n";
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string datasetName = argv[1];
    std::string weightsPath = argv[2];
    const std::string mode = argv[3];

    if (mode != "fresh" && mode != "resume")
    {
        std::cerr << "Invalid mode: " << mode << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    Dataset train;
    Dataset test;
    Cnn net;
    bool useDataAugmentation = true;
    AugmentationParameters dataAgumentRates;
    if (datasetName == "cifar10")
    {
        if (!loadCifar10Datasets(train, test))
        {
            std::cerr << "Could not load CIFAR-10.\n";
            return 1;
        }

        net = createCifar10Model();
        dataAgumentRates = {50,50,20,20,4};
    }
    else if (datasetName == "mnist")
    {
        if (!loadMnistDatasets(train, test))
        {
            std::cerr << "Could not load MNIST.\n";
            return 1;
        }
        net = createMnistModel();
        dataAgumentRates = {0,70,20,20,2};
    }
    else
    {
        std::cerr << "Unknown dataset: " << datasetName << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!validDataset(train) || !validDataset(test))
    {
        std::cerr << "Invalid dataset: the image and label counts differ.\n";
        return 1;
    }

    std::cout << "Dataset: " << datasetName << '\n'
              << "Training images: " << train.images.size() << '\n'
              << "Test images: " << test.images.size() << '\n';

    if (mode == "resume")
    {
        if (!validWeightsFile(weightsPath, net))
        {
            return 1;
        }

        net.load(weightsPath);
        std::cout << "Weights loaded from: " << weightsPath << '\n';
    }
    else
    {
        std::cout << "Model initialized with random weights.\n";
    }

    const int epochs = 0;
    const int batchSize = 64;
    const float learningRate = 0.05f;
    const float learningRateFactor = 0.90f;
    const float weightDecay = 0.0f;
    const int epochsBeforeSave = 1;

    net.train(train, epochs, batchSize, learningRate, learningRateFactor, weightDecay, useDataAugmentation, epochsBeforeSave, weightsPath, dataAgumentRates);

    std::cout << std::fixed << std::setprecision(2)
              << "Test accuracy: "
              << net.averageAccuracy(test.images, test.labels) << "%\n"
              << "Train accuracy: "
              << net.averageAccuracy(train.images, train.labels) << "%\n";

    return 0;
}
