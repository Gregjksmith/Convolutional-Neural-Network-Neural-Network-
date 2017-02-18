# Convolutional Neural Network and Fully Connected Neural Network

## Description
Presented is an open source implementation of a 2-layer fully connected Neural Network and a 6-layer Convolutional Neural Network. Tunable network parameters are learned using an input training set.

## API

### gs::NeuralNetwork
> 2-layer fully connected Neural Network. Network input size, output size and layer sizes are specified by the user.

```
gs::NeuralNetwork::NeuralNetwork(gs::ClContext* context, size_t inputSize, 
			size_t activationLayer1Size, size_t activationLayer2Size, 
			size_t outputLayerSize, float learningRate, 
			size_t minibatchSize, size_t epochs);
```

#### Parameters
**gs::ClContext* context**: *openCL context. Set context to NULL for automatic handling.*

**size_t inputSize**: *Number of input nodes.*

**size_t activationLayer1Size**: *Number hidden nodes in layer 1 of the network.*

**size_t activationLayer2Size**: *Number hidden nodes in layer 2 of the network.*

**size_t outputLayerSize**: *number of output nodes.*

**float learningRate**: *rate at which the gradient is traversed for parameter updates. Typically set between 1.0 and 10.0.*

**size_t minibatchSize**: *Number of random samples taken from the training set to perform updates. Smaller batch sizes have high update variance but may converge faster. For large training sets 200 is recommended.*

**size_t epochs**: *Total number of training iterations.*