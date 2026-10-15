#pragma once
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cnn/data/Dataset.h"

inline Dataset loadMnistCsv(const std::string &path, int maxRows = -1)
{
    Dataset ds;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Could not open MNIST file: " << path << '\n';
        return ds;
    }
    std::string line;
    int count = 0;
    int isfirstline = 1;
    while (std::getline(file, line))
    {
        if (maxRows != -1 && count >= maxRows)
        {
            break;
        }
        if (isfirstline == 1)
        {
            isfirstline = 0;
        }
        else
        {
            int val = 0;
            std::vector<float> pixels(28 * 28);
            int cnt = 0;
            for (size_t index = 2; index < line.size(); index++)
            {
                if (line[index] == ',')
                {
                    assert(cnt <= 28 * 28 - 1);
                    pixels[cnt] = val / 255.0f;
                    cnt++;
                    val = 0;
                }
                else
                {
                    val = val * 10 + line[index] - '0';
                }
            }
            assert(cnt <= 28 * 28 - 1);
            pixels[cnt] = val / 255.0f;
            ds.labels.push_back(line[0] - '0');
            ds.images.push_back(Tensor(1,28,28));
            ds.images.back().vals=pixels;
            count++;
        }
    }
    return ds;
}

inline bool loadMnistDatasets(Dataset &train, Dataset &test)
{
    train = loadMnistCsv("data/mnist/train.csv", 60000);
    test = loadMnistCsv("data/mnist/test.csv", 10000);

    return validDataset(train) && validDataset(test);
}
