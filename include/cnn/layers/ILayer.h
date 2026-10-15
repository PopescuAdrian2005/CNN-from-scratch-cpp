#pragma once
#include <fstream>
#include <memory>

#include "cnn/Tensor.h"

struct ILayer
{
    virtual int size() {return 0;}
    virtual void save(std::ofstream &) {};
    virtual void load(std::ifstream &) {};
    virtual void update(float , int , float ) {};
    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &nextlayer) = 0;
    virtual std::unique_ptr<ILayer> clone () = 0;
};

