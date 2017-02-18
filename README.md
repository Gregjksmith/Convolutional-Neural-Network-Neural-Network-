# Convolutional Neural Network and Fully Connected Neural Network

## Description
Presented is an open source implementation of a 2-layer fully connected Neural Network and a 6-layer Convolutional Neural Network. Tunable network parameters are learned using an input training set.

## API

### Neural Network

```
gs::NeuralNetwork::NeuralNetwork(gs::ClContext* context, size_t inputSize, 
			size_t activationLayer1Size, size_t activationLayer2Size, 
			size_t outputLayerSize, float learningRate, 
			size_t minibatchSize, size_t epochs);
```
> 2-layer fully connected Neural Network. Network input size, output size and layer sizes are specified by the user.
#### Parameters
+ **gs::ClContext* context**: *openCL context. Set context to NULL for automatic handling.*
+ **size_t inputSize**: *Number of input nodes.*
+ **size_t activationLayer1Size**: *Number hidden nodes in layer 1 of the network.*
+ **size_t activationLayer2Size**: *Number hidden nodes in layer 2 of the network.*
+ **size_t outputLayerSize**: *number of output nodes.*
+ **float learningRate**: *rate at which the gradient is traversed for parameter updates. Typically set between 1.0 and 10.0.*
+ **size_t minibatchSize**: *Number of random samples taken from the training set to perform updates. Smaller batch sizes have high update variance but may converge faster. For large training sets 200 is recommended.*
+ **size_t epochs**: *Total number of training iterations.*




```
void gs::NeuralNetwork::train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples);
```
> Trains the neural network parameters. The network parameters are set to best estimate the output samples with their corresponding input samples.
#### Parameters
+ **float* trainingSetInput**: *Input training set. Training samples must be vectorized. i.e. array size is (inputSize)x(numTrainingSamples).*
+ **float* trainingSetOutput**: *Output training set. Training samples must be vectorized. i.e. array size is (outputSize)x(numTrainingSamples).
+ **size_t numTrainingSamples**: *Number of training samples.*




```
void gs::NeuralNetwork::predict(float* inputVector, float* outputVector, size_t numSamples);
```
> Computes the output of the neural network for each of the input samples and places them in 'outputVector'.
#### Parameters
+ **float* inputVector**: *A vector containing 'numSamples' input samples. Array size must be (inputSize)x(numSamples).*  
+ **float* outputVector**: *Buffer used to place the output of the network. Array size must be (outputSize)X(numSamples).*
+ **size_t numSamples**: *Number of samples to be predicted.*




```
void gs::NeuralNetwork::exportNNParams(char* filePath);
```
> Writes the neural network parameters to a file located at 'filePath' 
#### Parameters
+ **char* filePath**: *Export file location.*




```
void gs::NeuralNetwork::importNNParams(char* filePath);
```
>inports the neural network parameters from 'filePath'
#### Parameters
+ **char* filePath**: *Export file location.*






### Convolutional Neural Network

```
gs::ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(ClContext* context, size_t inputImageWidth, 
		size_t inputImageHeight, size_t outputSize, 
		float learningRate, size_t minibatchSize, size_t numEpochs);
```
> 6-layer convolutional neural network, based on the LeNet-5 architecture. Inputs are images of size with specified width and height.
#### Parameters
+ **gs::ClContext* context**: *openCL context. Set context to NULL for automatic handling.*
+ **size_t inputImageWidth**: *Input image width.*
+ **size_t inputImageHeight**: *Input image height.*
+ **size_t outputSize**: *number of output nodes.*
+ **float learningRate**: *rate at which the gradient is traversed for parameter updates. Typically set between 1.0 and 10.0.*
+ **size_t minibatchSize**: *Number of random samples taken from the training set to perform updates. Smaller batch sizes have high update variance but may converge faster. For large training sets 200 is recommended.*
+ **size_t numEpochs**: *Total number of training iterations.*




```
void gs::ConvolutionalNeuralNetwork::train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples);
```
> Trains the neural network parameters. The network parameters are set to best estimate the output samples with their corresponding input samples.
#### Parameters
+ **float* trainingSetInput**: *Input training set. Training samples must be vectorized. i.e. array size is (inputSize)x(numTrainingSamples).*
+ **float* trainingSetOutput**: *Output training set. Training samples must be vectorized. i.e. array size is (outputSize)x(numTrainingSamples).
+ **size_t numTrainingSamples**: *Number of training samples.*




```
void gs::ConvolutionalNeuralNetwork::predict(float* inputVector, float* outputVector, size_t numSamples);
```
> Computes the output of the neural network for each of the input samples and places them in 'outputVector'.
#### Parameters
+ **float* inputVector**: *A vector containing 'numSamples' input samples. Array size must be (inputSize)x(numSamples).*  
+ **float* outputVector**: *Buffer used to place the output of the network. Array size must be (outputSize)X(numSamples).*
+ **size_t numSamples**: *Number of samples to be predicted.*




```
void gs::ConvolutionalNeuralNetwork::exportNNParams(char* filePath);
```
> Writes the neural network parameters to a file located at 'filePath' 
#### Parameters
+ **char* filePath**: *Export file location.*




```
void gs::ConvolutionalNeuralNetwork::importNNParams(char* filePath);
```
>inports the neural network parameters from 'filePath'
#### Parameters
+ **char* filePath**: *Export file location.*

## Example

```
#include "NeuralNetwork.h"
#include "CL\cl.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <opencv2\opencv.hpp>

#define NUM_TRAINING_IMAGES 60000
#define TRAINING_IMAGES_WIDTH 28
#define TRAINING_IMAGES_HEIGHT 28
#define TRAINING_SET_IMAGE_PATH "../Training Set/train-images.idx3-ubyte"
#define TRAINING_SET_LABEL_PATH "../Training Set/train-labels.idx1-ubyte"


#define NUM_TEST_IMAGES 10000
#define TEST_SET_IMAGE_PATH "../Training Set/t10k-images.idx3-ubyte"
#define TEST_SET_LABEL_PATH "../Training Set/t10k-labels.idx1-ubyte"

#define NN_PARAMS_EXPORT_PATH "../exports/exportNNParams.bin"
#define CNN_PARAMS_EXPORT_PATH "../exports/exportCNNParams.bin"


void loadTrainingSet(char* imageFilePath, char* labelFilePath, int trainingSetSize, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels);
float* createImageVector(std::vector<cv::Mat*> &inputImage);
float* createOutputVector(std::vector<unsigned char> &trainingLabels, size_t outputSize);
float errorRate(float* outputVector, std::vector<unsigned char> &trainingLabels, int vectorSize, size_t numSamples);

#define NN_LAYER_1_SIZE 50
#define NN_LAYER_2_SIZE 50
#define OUTPUT_SIZE 10
#define NN_LEARNING_RATE 3.0
#define MINI_BATCH_SIZE 200
#define NN_EPOCHS 5000

#define CNN_LEARNING_RATE 0.02
#define CNN_EPOCHS 100000

void main()
{
	vector<Mat*> trainingImages;
	vector<unsigned char> trainingLabels;

	/*load the training and test sets
		Assume functions are defined
	*/
	loadTrainingSet(TRAINING_SET_IMAGE_PATH, TRAINING_SET_LABEL_PATH, NUM_TRAINING_IMAGES, trainingImages, trainingLabels);
	
	/* Create the vectorized training sets */
	float* inputTrainingVector = createImageVector(trainingImages);
	float* outputTrainingVector = createOutputVector(trainingLabels, OUTPUT_SIZE);
	float testOutput[NUM_TRAINING_IMAGES * OUTPUT_SIZE];


	gs::ClContext* clContext = new gs::ClContext();

	/* Neural Network */
	gs::NeuralNetwork* nn = new gs::NeuralNetwork(clContext, (TRAINING_IMAGES_WIDTH * TRAINING_IMAGES_HEIGHT), 
									NN_LAYER_1_SIZE, NN_LAYER_2_SIZE, OUTPUT_SIZE, 
									NN_LEARNING_RATE, MINI_BATCH_SIZE, NN_EPOCHS);
	nn->train(inputTrainingVector, outputTrainingVector, NUM_TRAINING_IMAGES);
	nn->predict(inputTrainingVector, testOutput, NUM_TRAINING_IMAGES);
	float nnRecognitionError = errorRate(testOutput, trainingLabels, OUTPUT_SIZE, NUM_TRAINING_IMAGES);


	/* Convolutional Neural Network */
	gs::ConvolutionalNeuralNetwork* cnn = new gs::ConvolutionalNeuralNetwork(clContext,TRAINING_IMAGES_WIDTH,TRAINING_IMAGES_HEIGHT,
									OUTPUT_SIZE, CNN_LEARNING_RATE, MINI_BATCH_SIZE, CNN_EPOCHS);
	cnn->train(inputTrainingVector, outputTrainingVector, NUM_TRAINING_IMAGES);
	cnn->predict(inputTrainingVector, testOutput, NUM_TRAINING_IMAGES);
	float cnnRecognitionError = errorRate(testOutput, trainingLabels, OUTPUT_SIZE, NUM_TRAINING_IMAGES);

	delete inputTrainingVector;
	delete outputTrainingVector;
	trainingImages.clear();
	trainingLabels.clear();
	delete nn;
	delete cnn;
}
```