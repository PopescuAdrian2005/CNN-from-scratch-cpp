#pragma once
#include <algorithm>
#include <random>
#include <vector>
#include <cassert>
#include <array>

#include "cnn/Tensor.h"

using AugmentationParameters = std::array<int, 5>;

void flipHorizontal(Tensor &input)
{
    for (int c = 0; c < (int)input.channels; c++)
    {
        for (int i = 0; i < (int)input.rows; i++)
        {
            for (int j = 0; j < (int)input.cols / 2; j++)
            {
                std::swap(input.at(c, i, j), input.at(c, i, input.cols - 1 - j));
            }
        }
    }
}

void randomBrightness(Tensor &image, float strength = 0.1f)
{
    std::uniform_real_distribution<float> dist(
        1.0f - strength,
        1.0f + strength);

    const float factor = dist(gen);

    for (float &pixel : image.vals)
        pixel = std::clamp(pixel * factor, 0.0f, 1.0f);
}

void randomContrast(Tensor &image, float strength = 0.1f)
{
    float mean = 0.0f;

    for (float pixel : image.vals)
        mean += pixel;

    mean /= image.vals.size();

    std::uniform_real_distribution<float> dist(
        1.0f - strength,
        1.0f + strength);

    const float factor = dist(gen);

    for (float &pixel : image.vals)
    {
        pixel = std::clamp(mean + factor * (pixel - mean),0.0f,1.0f);
    }
}
void randomOffset(Tensor &image, int padding)
{
    Tensor out(image, padding);
    std::uniform_int_distribution<int> offsetDist(0, 2 * padding);
    int x = offsetDist(gen);
    int y = offsetDist(gen);
    for (int c = 0; c < image.channels; c++)
    {
        for (int i = x; i < x + image.rows; i++)
        {
            for (int j = y; j < y + image.cols; j++)
            {
                image.at(c,i-x,j-y)=out.at(c,i,j);
            }
        }
    }
}
Tensor augmentDataset(const Tensor &input,AugmentationParameters &dataAgumentRates)
{
    Tensor output = input;
    std::uniform_int_distribution<int> chance(0, 99);
    if (chance(gen) < dataAgumentRates[0])
    {
        flipHorizontal(output);
    }
    if (chance(gen) < dataAgumentRates[1])
    {
        randomOffset(output, dataAgumentRates[4]);
    }
    if (chance(gen) < dataAgumentRates[2])
    {
        randomBrightness(output, 0.1f);
    }
    if (chance(gen) < dataAgumentRates[3])
    {
        randomContrast(output, 0.1f);
    }
    return output;
}
std::vector<Tensor> ModifyData(const std::vector<Tensor> &input,AugmentationParameters &dataAgumentRates)
{
    std::vector<Tensor> output(input.size());
    for (int i = 0; i < (int)input.size(); i++)
    {
        output[i] = augmentDataset(input[i],dataAgumentRates);
    }
    return output;
}

void shuffleDataset(
    std::vector<Tensor>& inputs,
    std::vector<int>& labels)
{
    assert(inputs.size() == labels.size());

    static std::mt19937 generator(std::random_device{}());

    for (size_t remaining = inputs.size();remaining > 1;remaining--)
    {
        std::uniform_int_distribution<size_t> distribution(0,remaining - 1);

        size_t randomIndex = distribution(generator);
        size_t lastIndex = remaining - 1;

        std::swap(inputs[lastIndex], inputs[randomIndex]);
        std::swap(labels[lastIndex], labels[randomIndex]);
    }
}