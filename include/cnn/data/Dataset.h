#pragma once
#include <vector>

#include "cnn/Tensor.h"
struct Dataset
{
    std::vector<Tensor> images;
    std::vector<int> labels;
};

inline bool validDataset(const Dataset &dataset)
{
    return (!dataset.images.empty() && (dataset.images.size() == dataset.labels.size()));
}
