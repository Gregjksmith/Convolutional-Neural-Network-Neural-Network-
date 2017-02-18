# Convolutional Neural Network and Fully Connected Neural Network

## Description
Presented is an open source implementation of a 2-layer fully connected Neural Network and a 6-layer Convolutional Neural Network. Tunable network parameters are learned using an input training set.

## API


```
gs::NeuralNetwork::NeuralNetwork(gs::ClContext* context, size_t inputSize, 
			size_t activationLayer1Size, size_t activationLayer2Size, 
			size_t outputLayerSize, float learningRate, 
			size_t minibatchSize, size_t epochs);
```
> 2-layer fully connected Neural Network. Network input size, output size and layer sizes are specified by the user.
#### Parameters
**gs::ClContext* context**: *openCL context. Set context to NULL for automatic handling.*
**size_t inputSize**: *Number of input nodes.*
**size_t activationLayer1Size**: *Number hidden nodes in layer 1 of the network.*
**size_t activationLayer2Size**: *Number hidden nodes in layer 2 of the network.*
**size_t outputLayerSize**: *number of output nodes.*
**float learningRate**: *rate at which the gradient is traversed for parameter updates. Typically set between 1.0 and 10.0.*
**size_t minibatchSize**: *Number of random samples taken from the training set to perform updates. Smaller batch sizes have high update variance but may converge faster. For large training sets 200 is recommended.*
**size_t epochs**: *Total number of training iterations.*




```
void gs::NeuralNetwork::train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples);
```
> Trains the neural network parameters. The network parameters are set to best estimate the output samples with their corresponding input samples.
#### Parameters
**float* trainingSetInput**: *Input training set. Training samples must be vectorized. i.e. array size is (inputSize)x(numTrainingSamples).*
**float* trainingSetOutput**: *Iutput training set. Training samples must be vectorized. i.e. array size is (outputSize)x(numTrainingSamples).
**size_t numTrainingSamples**: *Number of training samples.*




```
void gs::NeuralNetwork::predict(float* inputVector, float* outputVector, size_t numSamples);
```
> Computes the output of the neural network for each of the input samples and places them in 'outputVector'.
#### Parameters
**float* inputVector**: *A vector containing 'numSamples' input samples. Array size must be (inputSize)x(numSamples).*  
**float* outputVector**: *Buffer used to place the output of the network. Array size must be (outputSize)X(numSamples).*
**size_t numSamples**: *Number of samples to be predicted.*




```
void gs::NeuralNetwork::exportNNParams(char* filePath);
```
> Writes the neural network parameters to a file located at 'filePath' 
#### Parameters
**char* filePath**:*Export file location.*




```
void gs::NeuralNetwork::importNNParams(char* filePath);
```
>inports the neural network parameters from 'filePath'
#### Parameters
**char* filePath**:*Export file location.*