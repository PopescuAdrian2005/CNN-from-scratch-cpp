#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "cnn/data/Dataset.h"


inline Dataset loadCifar10(const std::string &path, int maxRows = -1)
{
    Dataset ds;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open CIFAR-10 file: " << path << '\n';
        return ds;
    }
    for (int row = 0; row < maxRows; row++)
    {
        unsigned char x;
        file.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
        ds.labels.push_back(x);
        ds.images.push_back(Tensor(3,32,32));
        for (int z = 0; z < 3; z++)
        {
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    unsigned char x;
                    file.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
                    ds.images[row].at(z,i,j)=x/255.0f;
                }
            }
        }
    }
    return ds;
}

inline void appendCifar10(const std::string &path, int maxRows, Dataset &ds)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open CIFAR-10 file: " << path << '\n';
        return;
    }
    int offset = ds.images.size();
    for (int row = 0; row < maxRows; row++)
    {
        unsigned char x;
        file.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
        ds.labels.push_back(x);
        ds.images.push_back(Tensor(3,32,32));
        for (int z = 0; z < 3; z++)
        {
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    unsigned char x;
                    file.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
                    ds.images[row+offset].at(z,i,j)=x/255.0f;
                }
            }
        }
    }
}

inline bool loadCifar10Datasets(Dataset &train, Dataset &test)
{
    constexpr int imagesPerFile = 10000;
    const std::string directory = "data/cifar10/";

    train = loadCifar10(directory + "data_batch_1.bin", imagesPerFile);
    appendCifar10(directory + "data_batch_2.bin", imagesPerFile, train);
    appendCifar10(directory + "data_batch_3.bin", imagesPerFile, train);
    appendCifar10(directory + "data_batch_4.bin", imagesPerFile, train);
    appendCifar10(directory + "data_batch_5.bin", imagesPerFile, train);
    test = loadCifar10(directory + "test_batch.bin", 10000);

    return validDataset(train) && validDataset(test);
}
