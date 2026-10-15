# CNN from Scratch in C++

A small convolutional neural network implementation written from scratch in C++17. The project implements tensors, convolutional and dense layers, activation functions, pooling, backpropagation, data augmentation, training, evaluation, and binary weight persistence without using a machine-learning framework.

## Supported datasets

- MNIST: 28 x 28 grayscale images and 10 classes.
- CIFAR-10: 32 x 32 RGB images and 10 classes.

The dataset files are not intended to be stored in Git. The program currently expects this local layout:

```text
data/
├── mnist/
│   ├── train.csv
│   └── test.csv
└── cifar10/
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── test_batch.bin
    └── batches.meta.txt
```

## Project structure

```text
include/cnn/
├── Cnn.h
├── Matrix.h
├── Softmax.h
├── Tensor.h
├── data/
│   ├── Cifar10Loader.h
│   ├── Dataset.h
│   └── MnistLoader.h
├── layers/
│   ├── ConvLayer.h
│   ├── DenseLayer.h
│   ├── FlattenLayer.h
│   ├── ILayer.h
│   ├── MaxPoolLayer.h
│   └── ReluLayer.h
└── training/
    └── DataAugmentation.h

src/
└── main.cpp
```

The MNIST and CIFAR-10 model architectures are defined in `src/main.cpp`. Dataset parsing and the default dataset paths are kept in their corresponding loader headers.

## Model architectures

Both models use stride-1 convolutions with 3 x 3 kernels and one-pixel zero padding, so convolutional layers preserve the spatial resolution. Every convolution is followed by ReLU. `MaxPoolLayer` uses a fixed 2 x 2 window with stride 2 and therefore halves the height and width.

### CIFAR-10

Input shape: `3 x 32 x 32` (RGB).

| Stage | Operation | Output shape |
|---|---|---|
| Input | CIFAR-10 image | `3 x 32 x 32` |
| Block 1 | Conv `3 -> 16`, ReLU, Conv `16 -> 16`, ReLU | `16 x 32 x 32` |
| Pool 1 | 2 x 2 max pooling | `16 x 16 x 16` |
| Block 2 | Conv `16 -> 32`, ReLU, Conv `32 -> 32`, ReLU | `32 x 16 x 16` |
| Pool 2 | 2 x 2 max pooling | `32 x 8 x 8` |
| Block 3 | Conv `32 -> 64`, ReLU, Conv `64 -> 64`, ReLU | `64 x 8 x 8` |
| Pool 3 | 2 x 2 max pooling | `64 x 4 x 4` |
| Classifier | Flatten, Dense `1024 -> 128`, ReLU, Dense `128 -> 10` | `10` logits |

The CIFAR-10 model has **204,570 trainable parameters**.

### MNIST

Input shape: `1 x 28 x 28` (grayscale).

| Stage | Operation | Output shape |
|---|---|---|
| Input | MNIST image | `1 x 28 x 28` |
| Block 1 | Conv `1 -> 8`, ReLU, Conv `8 -> 8`, ReLU | `8 x 28 x 28` |
| Pool 1 | 2 x 2 max pooling | `8 x 14 x 14` |
| Block 2 | Conv `8 -> 16`, ReLU, Conv `16 -> 16`, ReLU | `16 x 14 x 14` |
| Pool 2 | 2 x 2 max pooling | `16 x 7 x 7` |
| Classifier | Flatten, Dense `784 -> 128`, ReLU, Dense `128 -> 10` | `10` logits |

The MNIST model has **105,922 trainable parameters**.

Softmax is applied by `Cnn::forward` after the final dense layer. Training uses softmax cross-entropy and mini-batch gradient descent.

## Achieved results

The supplied weight files produce the following results on the complete training and test sets:

| Dataset | Training images | Test images | Train accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| CIFAR-10 | 50,000 | 10,000 | **82.32%** | **76.30%** |
| MNIST | 60,000 | 10,000 | **99.43%** | **99.08%** |

These values were measured by loading `weights/cifar10.bin` and `weights/mnist.bin` in `resume` mode with `epochs` set to `0`. In that configuration, no additional training takes place; the program only evaluates the loaded model. Results may vary after retraining because weight initialization, shuffling, and data augmentation are random.

## Requirements

- A C++17 compiler such as GCC.
- OpenMP support.
- `taskset`, `nice`, and `systemd-run` are optional Linux utilities used by some commands in `command.txt`.

## Usage

```text
./cnn <dataset> <weights-file> <fresh|resume>
```

`fresh` starts with randomly initialized weights. `resume` validates and loads the supplied weight file before continuing. Weight files are architecture-specific and are rejected when their size does not match the selected model or when they contain non-finite values.

## Training configuration

Training parameters are currently configured near the end of `src/main.cpp`:

```cpp
const int epochs = 0;
const int batchSize = 64;
const float learningRate = 0.05f;
const float learningRateFactor = 0.90f;
const float weightDecay = 0.0f;
const int epochsBeforeSave = 1;
```

Set `epochs` to a positive value to train. With `epochs == 0`, the program only evaluates the initialized or loaded model.

The five augmentation values are, in order, the percentage chances for horizontal flipping, translation, brightness adjustment, and contrast adjustment, followed by the translation padding in pixels:

```cpp
// flip %, translation %, brightness %, contrast %, translation padding
dataAgumentRates = {50, 50, 20, 20, 4}; // CIFAR-10
dataAgumentRates = { 0, 70, 20, 20, 2}; // MNIST
```

Horizontal flipping is deliberately disabled for MNIST. Set `useDataAugmentation` to `false` to train on unmodified images.

## Changing the architecture

Edit `createCifar10Model()` or `createMnistModel()` in `src/main.cpp`. Layers are executed in the order in which they are appended to `net.layers`.

The convolution constructor is:

```cpp
ConvLayer(outputChannels, inputChannels, kernelRows, kernelColumns, padding)
```

When changing a model:

1. Keep adjacent channel counts consistent. For example, a convolution producing 32 channels must be followed by a convolution whose `inputChannels` is 32.
2. Recalculate the first dense layer's input size. With same-padded convolutions, each `MaxPoolLayer` halves both spatial dimensions. The flattened size is `channels * height * width` after the last pooling layer.
3. Keep the final dense layer at 10 outputs for MNIST and CIFAR-10, because both datasets have 10 classes.
4. Rebuild the executable.
5. Start the changed model with `fresh` and preferably use a new weight-file name. Existing binary weights are tied to the exact sequence and sizes of parameterized layers.

For example, to widen the second MNIST block from 16 to 32 channels, change its two convolutions to `ConvLayer(32, 8, 3, 3, 1)` and `ConvLayer(32, 32, 3, 3, 1)`. If the number of pooling stages remains unchanged, the classifier must then begin with `DenseLayer(7 * 7 * 32, 128)`.

## Notes about weight files

Weight files contain raw binary `float` values. Move or copy them with filesystem commands such as `mv` or `cp`; do not open and copy their contents through a text editor.

The current model sizes are:

- CIFAR-10: 818,280 bytes.
- MNIST: 423,688 bytes.

Changing the architecture changes the expected weight-file size and makes old weights incompatible.
