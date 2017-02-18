/* NEURAL NETWORK KERNEL 

 AUTHOR: Greg Smith, 2016

OpenCL Neural Network kernels.
Trains a neural network to minimize the L2 norm error between f(x) and the output vector.
x is the input vector, N samples where each sample is size 'INPUT_SIZE'.
f(x) is the neural network output.
	2 Layers, layers are of size 'LAYER_1_SIZE' and 'LAYER_2_SIZE' respectively.
output vector is N samples of 'OUTPUT_SIZE' size.

*/

#ifndef INPUT_SIZE
#define INPUT_SIZE (28*28)
#endif

#ifndef LAYER_1_SIZE
#define LAYER_1_SIZE 50
#endif

#ifndef LAYER_2_SIZE
#define LAYER_2_SIZE 50
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 10
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 3.0
#endif

#ifndef STOCHASTIC_SAMPLING_SIZE
#define STOCHASTIC_SAMPLING_SIZE 200
#endif

#define ACTIVATION_SIZE (LAYER_1_SIZE + LAYER_2_SIZE + OUTPUT_SIZE)

/* WEIGHT STRUCTURE */
typedef struct _NNweights
{
	float layer1Weights[INPUT_SIZE*LAYER_1_SIZE];
	float layer1Bias[LAYER_1_SIZE];

	float layer2Weights[LAYER_1_SIZE*LAYER_2_SIZE];
	float layer2Bias[LAYER_2_SIZE];

	float layerOutputWeights[LAYER_2_SIZE*OUTPUT_SIZE];
	float layerOutputBias[OUTPUT_SIZE];
}NNweights;

/* ACTIVATION STRUCTURE */
typedef struct _Activation
{
	float layer1Activation[LAYER_1_SIZE];
	float layer2Activation[LAYER_2_SIZE];
	float layerOutputActivation[OUTPUT_SIZE];
}Activation;

/* ACTIVATION DELTA STRUCTURE */
typedef struct _ActivationDelta
{
	float layer1ActivationDelta[LAYER_1_SIZE];
	float layer2ActivationDelta[LAYER_2_SIZE];
	float layerOutputActivationDelta[OUTPUT_SIZE];
}ActivationDelta;

/* GRADIENT STRUCTURE */
typedef struct _Gradient
{
	float layer1Weights[INPUT_SIZE*LAYER_1_SIZE];
	float layer1Bias[LAYER_1_SIZE];

	float layer2Weights[LAYER_1_SIZE*LAYER_2_SIZE];
	float layer2Bias[LAYER_2_SIZE];

	float layerOutputWeights[LAYER_2_SIZE*OUTPUT_SIZE];
	float layerOutputBias[OUTPUT_SIZE];
}Gradient;

/* ================================== HEADERS ==========================================*/

/*
float sampleImage
gets the input image pixel intensity from the input image vector.
@param float* input : input image vector.
@param int imageIndex : index of the image in the image vector.
@param int pixelIndex : index of the pixel within an input image.
*/
float sampleImage(__global float* input, int imageIndex, int pixelIndex);

/*
float sampleOutput
samples the output vector.
@param float* output : output vector.
@param int outputIndex : output vector index.
@param int layerIndex : activation index within the output layer.
*/
float sampleOutput(__global float* output, int outputIndex, int layerIndex);

/*
void getWeightIndexLayer1
gets the input index and layer1Index from the corresponding global weight index.
@param int weightIndex : global vector weight index.
@param int* inputIndex : input activation index.
@param int* layer1Index : layer 1 activation index.  
*/
void getWeightIndexLayer1(int weightIndex, int* inputIndex, int* layer1Index);

/*
void getWeightIndexLayer2
gets the layer1 index and the layer 2 index from the global weight index.
@param weightIndex : global vector weight index.
@param int* layer1Index : layer 1 activation index.
@param int* layer2Index : layer 2 activation index.
*/
void getWeightIndexLayer2(int weightIndex, int* layer1Index, int* layer2Index);

/*
void getWeightIndexOutput
gets the layer2 index and the output index from the global weight index.
@param weightIndex : global vector weight index.
@param int* layer2Index : layer 2 activation index.
@param int* outputIndex : output activation index.
*/
void getWeightIndexOutput(int weightIndex, int* layer2Index, int* outputIndex);

/*
float getWeightLayer1
gets the NN weight connecting the inputIndex and layer1Index.
@param NNweight* weight : NN weight vector
@param int inputIndex : input activation index.
@param int layer1Index : layer 1 activation index.
*/
float getWeightLayer1(__global NNweights* weights, int inputIndex, int layer1Index);

/*
void getBiasLayer1
gets the bias at a layer 1 activation.
@param NNweight* weight : NN weight vector
@param int* layer1Index : layer 1 activation index.
*/
float getBiasLayer1(__global NNweights* weights, int layer1Index);


/*
float getWeightLayer2
gets the NN weight connecting the layer1Index and layer2Index.
@param NNweight* weight : NN weight vector
@param int layer1Index : layer 1 activation index.
@param int layer2Index : layer 2 activation index.
*/
float getWeightLayer2(__global NNweights* weights, int layer1Index, int layer2Index);

/*
void getBiasLayer2
gets the bias at a layer 2 activation.
@param NNweight* weight : NN weight vector
@param int* layer2Index : layer 2 activation index.
*/
float getBiasLayer2(__global NNweights* weights, int layer2Index);

/*
float getWeightLayerOutput
gets the NN weight connecting the layer2Index and layerOutputIndex.
@param NNweight* weight : NN weight vector
@param int layer2Index : layer 2 activation index.
@param int layerOutputIndex : layer output activation index.
*/
float getWeightLayerOutput(__global NNweights* weights, int layer2Index, int layerOutputIndex);

/*
float getBiasLayerOutput
gets the NN bias at an output layer index.
@param NNweight* weight : NN weight vector
@param int layerOutputIndex : layer output activation index.
*/
float getBiasLayerOutput(__global NNweights* weights, int layerOutputIndex);

/*
float getGradientWeightLayer1
samples the weight gradient from the gradient vector in layer 1.
@param int inputIndex
@param int layer1Index
*/
float getGradientWeightLayer1(__global Gradient* gradient, int inputIndex, int layer1Index);

/*
float getGradientBiasLayer1
samples the bias gradient from the gradient vector in layer 1.
@param int layer1Index
*/
float getGradientBiasLayer1(__global Gradient* gradient, int layer1Index);

/*
float getGradientWeightLayer2
samples the weight gradient from the gradient vector in layer 2.
@param int layer1Index
*/
float getGradientWeightLayer2(__global Gradient* gradient, int layer1Index, int layer2Index);

/*
float getGradientBiasLayer1
samples the bias gradient from the gradient vector in layer 2.
@param int layer2Index
*/
float getGradientBiasLayer2(__global Gradient* gradient, int layer2Index);

/*
float getGradientWeightLayerOutput
samples the weight gradient from the gradient vector in layer output.
@param int layer2Index
@param int layerOutputIndex
*/
float getGradientWeightLayerOutput(__global Gradient* gradient, int layer2Index, int layerOutputIndex);

/*
float getGradientBiasLayerOutput
samples the bias gradient from the gradient vector in layer output.
@param int layerOutputIndex
*/
float getGradientBiasLayerOutput(__global Gradient* gradient, int layerOutputIndex);

/*
float activationFunction
sigmoid function.
@param float x
*/
float activationFunction(float x);

/*
float activation Derivative
derivative of 'activationFunction' evaluated at x
@param float x
*/
float activationDerivative(float x);

/*
float tanHyperbolicDerivative
derivative of tanh() evaluated at x
@param float x
*/
float tanHyperbolicDerivative(float x);



/*
kernel void activationLayer1
computes the activations at layer 1
@param float* input : input vector
@param int inputIndex
@param Activation* activation
@param NNweights* weights
*/
__kernel void activationLayer1(__global float* input, __global int* inputIndex, __global Activation* activation, __global NNweights* weights);

/*
kernel void activationLayer2
computes the activations at layer 2
@param Activation* activation
@param NNweights* weights
*/
__kernel void activationLayer2(__global Activation* activation, __global NNweights* weights);

/*
kernel void activationLayerOutput
computes the activations at the output layer
@param Activation* activation
@param NNeights* weights
*/
__kernel void activationLayerOutput(__global Activation* activation, __global NNweights* weights);

/*
kernel void activationOutputDelta
computes the activations deltas (the derivative of the output w.r.t the activation) at the output layer.
@param float* output : output vector
@param int outputIndex
@param Activation* activation
@param ActivationDelta* activationDelta
*/
__kernel void activationOutputDelta(__global float* output, __global int* outputIndex, __global Activation* activation, __global ActivationDelta* activationDelta);

/*
kernel void activationLayer2Delta
computes the activations deltas (the derivative of the output w.r.t the activation) at layer 2.
@param Activation* activation
@param ActivationDelta* activationDelta
@param NNweights* weights
*/
__kernel void activationLayer2Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights);

/*
kernel void activationLayer1Delta
computes the activations deltas (the derivative of the output w.r.t the activation) at layer 1.
@param Activation* activation
@param ActivationDelta* activationDelta
@param NNweights* weights
*/
__kernel void activationLayer1Delta( __global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights);



/*
kernel void addGradientWeightLayer1
computes the weight in layer 1gradient and adds it to the gradient vector.
@param float* input
@param int inputIndex
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayer1(__global float* input, __global int* inputIndex, __global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientWeightLayer2
computes the weight gradient in layer 2and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientWeightLayerOutput
computes the weight gradient in the output layer and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayer1
computes the bias gradient in layer 1 and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayer1(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayer2
computes the bias gradient in layer 2 and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayerOutput
computes the bias gradient in the output layer and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void normalizeGradient
divides the gradient by the number of samples.
@param float* gradient
@param int numSamples
*/
__kernel void normalizeGradient(__global float* gradient, int numSamples);

/*
kernel void updateNNparams
updates the Neural netowrk weights by the learningRate times the negative gradient
@param float* weights
@param float* gradient
*/
__kernel void updateNNparams(__global float* weights, __global float* gradient);


/*
kernel void cost
Computes the L2 norm between the output activations and the outputVector at index outputIndex. Cost is divided by 2, such that
the gradient is not scales by 2. The cost is then added to the buffer returnCost
@param float* outputVector
@param int outputIndex
@param Activation* activation
@param float* returnCost : return cost is a floating point buffer of size 1, used to compute the average cost over N samples.
*/
__kernel void cost(__global float* outputVector, __global int* outputIndex, __global Activation* activation, __global float* returnCost);



/* ==================================================================================== */

float sampleImage(__global float* input, int imageIndex, int pixelIndex)
{
	int i = imageIndex*INPUT_SIZE + pixelIndex;
	return input[i];
}

float sampleOutput(__global float* output, int outputIndex, int layerIndex)
{
	int i = outputIndex*OUTPUT_SIZE + layerIndex;
	return output[i];
}

float getWeightLayer1(__global NNweights* weights, int inputIndex, int layer1Index)
{
	int i = inputIndex*LAYER_1_SIZE + layer1Index;
	return weights->layer1Weights[i];
}

float getBiasLayer1(__global NNweights* weights, int layer1Index)
{
	return weights->layer1Bias[layer1Index];
}

float getWeightLayer2(__global NNweights* weights, int layer1Index, int layer2Index)
{
	int i = layer1Index*LAYER_2_SIZE + layer2Index;
	return weights->layer2Weights[i];
}

float getBiasLayer2(__global NNweights* weights, int layer2Index)
{
	return weights->layer2Bias[layer2Index];
}

float getWeightLayerOutput(__global NNweights* weights, int layer2Index, int layerOutputIndex)
{
	int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;
	return weights->layerOutputWeights[i];
}

float getBiasLayerOutput(__global NNweights* weights, int layerOutputIndex)
{
	return weights->layerOutputBias[layerOutputIndex];
}


float getGradientWeightLayer1(__global Gradient* gradient, int inputIndex, int layer1Index)
{
	int i = inputIndex*LAYER_1_SIZE + layer1Index;
	return gradient->layer1Weights[i];
}
float getGradientBiasLayer1(__global Gradient* gradient, int layer1Index)
{
	return gradient->layer1Bias[layer1Index];
}

float getGradientWeightLayer2(__global Gradient* gradient, int layer1Index, int layer2Index)
{
	int i = layer1Index*LAYER_2_SIZE + layer2Index;
	return gradient->layer2Weights[i];
}
float getGradientBiasLayer2(__global Gradient* gradient, int layer2Index)
{
	return gradient->layer2Bias[layer2Index];
}

float getGradientWeightLayerOutput(__global Gradient* gradient, int layer2Index, int layerOutputIndex)
{
	int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;
	return gradient->layerOutputWeights[i];
}
float getGradientBiasLayerOutput(__global Gradient* gradient, int layerOutputIndex)
{
	return gradient->layerOutputBias[layerOutputIndex];
}

float activationFunction(float x)
{
	float s = 1.0/(1.0 + exp(-x));
	return s;
}

float activationDerivative(float x)
{
	float s = x;//activationFunction(x);
	s = s*(1.0 - s);
	return s;
}

float tanHyperbolicDerivative(float x)
{
	float tanhResult = tanh(x);
	return (1.0 - tanhResult*tanhResult);
}

__kernel void activationLayer1(__global float* input, __global int* inputIndex, __global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % LAYER_1_SIZE;
	int activationId_y = activationId / LAYER_1_SIZE;
	int ix = inputIndex[activationId_y];

	float activationSum = 0.0;
	for(int pixelIndex = 0; pixelIndex < INPUT_SIZE; pixelIndex++)
	{
		float im = sampleImage(input, ix, pixelIndex);
		float w = getWeightLayer1(weights, pixelIndex, activationId_x);
		activationSum += im*w;
	}
	activationSum += getBiasLayer1(weights, activationId_x);
	activationSum = activationFunction(activationSum);
	(activation[activationId_y]).layer1Activation[activationId_x] = activationSum;
}

__kernel void activationLayer2(__global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % LAYER_2_SIZE;
	int activationId_y = activationId / LAYER_2_SIZE;

	float activationSum = 0.0;
	for(int layer1Index = 0; layer1Index < LAYER_1_SIZE; layer1Index++)
	{
		float act1 = (activation[activationId_y]).layer1Activation[layer1Index];
		float w = getWeightLayer2(weights, layer1Index, activationId_x);
		activationSum += act1*w;
	}
	activationSum += getBiasLayer2(weights, activationId_x);
	activationSum = activationFunction(activationSum);
	(activation[activationId_y]).layer2Activation[activationId_x] = activationSum;
}

__kernel void activationLayerOutput(__global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % OUTPUT_SIZE;
	int activationId_y = activationId / OUTPUT_SIZE;

	float activationSum = 0.0;
	for(int layer2Index = 0; layer2Index < LAYER_2_SIZE; layer2Index++)
	{
		float act2 = (activation[activationId_y]).layer2Activation[layer2Index];
		float w = getWeightLayerOutput(weights, layer2Index, activationId_x);
		activationSum += act2*w;
	}
	activationSum += getBiasLayerOutput(weights, activationId_x);
	activationSum = activationFunction(activationSum);
	(activation[activationId_y]).layerOutputActivation[activationId_x] = activationSum;
}




__kernel void activationOutputDelta(__global float* output, __global int* outputIndex, __global Activation* activation, __global ActivationDelta* activationDelta)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % OUTPUT_SIZE;
	int activationId_y = activationId / OUTPUT_SIZE;
	int oi = outputIndex[activationId_y];

	float delta = 0.0;
	float activationSample = (activation[activationId_y]).layerOutputActivation[activationId_x];
	delta = (activationSample - sampleOutput(output, oi, activationId_x));
	(activationDelta[activationId_y]).layerOutputActivationDelta[activationId_x] = delta;
}

__kernel void activationLayer2Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % LAYER_2_SIZE;
	int activationId_y = activationId / LAYER_2_SIZE;

	float delta = 0.0;
	float activationDeltaSample;
	float activationSample;
	float weightSample;

	for(int outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++)
	{
		activationSample = (activation[activationId_y]).layerOutputActivation[outputIndex];
		activationDeltaSample = (activationDelta[activationId_y]).layerOutputActivationDelta[outputIndex];
		weightSample = getWeightLayerOutput(weights, activationId_x, outputIndex);
		delta += activationDeltaSample*weightSample*activationDerivative(activationSample);
	}
	(activationDelta[activationId_y]).layer2ActivationDelta[activationId_x] = delta;
}

__kernel void activationLayer1Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	int activationId_x = activationId % LAYER_1_SIZE;
	int activationId_y = activationId / LAYER_1_SIZE;

	float delta = 0.0;
	float activationSample;
	float activationDeltaSample;
	float weightSample;

	for(int layer2Index = 0; layer2Index < LAYER_2_SIZE; layer2Index++)
	{
		activationDeltaSample = (activationDelta[activationId_y]).layer2ActivationDelta[layer2Index];
		activationSample = (activation[activationId_y]).layer2Activation[layer2Index];
		weightSample = getWeightLayer2(weights, activationId_x, layer2Index);
		delta += activationDeltaSample*weightSample*activationDerivative(activationSample);
	}

	(activationDelta[activationId_y]).layer1ActivationDelta[activationId_x] = delta;
}

void getWeightIndexLayer1(int weightIndex, int* inputIndex, int* layer1Index)
{
	//int i = inputIndex*LAYER_1_SIZE + layer1Index;

	*inputIndex = weightIndex / LAYER_1_SIZE;
	*layer1Index = weightIndex - ((*inputIndex)*LAYER_1_SIZE);
}

void getWeightIndexLayer2(int weightIndex, int* layer1Index, int* layer2Index)
{
	//int i = layer1Index*LAYER_2_SIZE + layer2Index;

	*layer1Index = weightIndex / LAYER_2_SIZE;
	*layer2Index = weightIndex - ((*layer1Index)*LAYER_2_SIZE);
}

void getWeightIndexOutput(int weightIndex, int* layer2Index, int* outputIndex)
{
	//int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;

	*layer2Index = weightIndex / OUTPUT_SIZE;
	*outputIndex = weightIndex - ((*layer2Index)*OUTPUT_SIZE);
}



__kernel void addGradientWeightLayer1(__global float* input, __global int* inputIndex, __global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	int oi;
	float grad = gradient->layer1Weights[gradientId];
	for(int i = 0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		oi = inputIndex[i];
		int layer1ActivationIndex;
		int inputActivation;
		getWeightIndexLayer1(gradientId, &inputActivation, &layer1ActivationIndex);

		float inSample = sampleImage(input, oi, inputActivation);
		float del1 = (activationDelta[i]).layer1ActivationDelta[layer1ActivationIndex];
		float act1 = (activation[i]).layer1Activation[layer1ActivationIndex];

		grad = grad + del1*inSample*activationDerivative(act1);
	}
	gradient->layer1Weights[gradientId] = grad;
}

__kernel void addGradientWeightLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	float grad = gradient->layer2Weights[gradientId];
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		int layer2ActivationIndex;
		int layer1ActivationIndex;
		getWeightIndexLayer2(gradientId, &layer1ActivationIndex, &layer2ActivationIndex);
		float al1 = (activation[i]).layer1Activation[layer1ActivationIndex];
		float del2 = (activationDelta[i]).layer2ActivationDelta[layer2ActivationIndex];
		float act2 = (activation[i]).layer2Activation[layer2ActivationIndex];
		grad = grad + del2*al1*activationDerivative(act2);
	}
	gradient->layer2Weights[gradientId] = grad;
}

__kernel void addGradientWeightLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);

	int outputActivationIndex;
	int layer2ActivationIndex;
	getWeightIndexOutput(gradientId, &layer2ActivationIndex, &outputActivationIndex);
		
	float grad = gradient->layerOutputWeights[gradientId];
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float al2 = (activation[i]).layer2Activation[layer2ActivationIndex];
		float delOut = (activationDelta[i]).layerOutputActivationDelta[outputActivationIndex];
		float actOut = (activation[i]).layerOutputActivation[outputActivationIndex];
		grad = grad + delOut*al2*activationDerivative(actOut);

		// gradient = gradient + activationLayer2*deltaOutput
	}
	gradient->layerOutputWeights[gradientId] = grad;
}

__kernel void addGradientBiasLayer1(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	float grad = gradient->layer1Bias[gradientId];
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float grad = gradient->layer1Bias[gradientId];
		float actDelta = (activationDelta[i]).layer1ActivationDelta[gradientId];
		float act = (activation[i]).layer1Activation[gradientId];
		grad = grad + actDelta*activationDerivative(act);
	}
	gradient->layer1Bias[gradientId] = grad;
}

__kernel void addGradientBiasLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	float grad = gradient->layer2Bias[gradientId];
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float actDelta = (activationDelta[i]).layer2ActivationDelta[gradientId];
		float act = (activation[i]).layer2Activation[gradientId];
		grad = grad + actDelta*activationDerivative(act);
	}
	gradient->layer2Bias[gradientId] = grad;
}

__kernel void addGradientBiasLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	float grad = gradient->layerOutputBias[gradientId];
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float actDelta = (activationDelta[i]).layerOutputActivationDelta[gradientId];
		float act = (activation[i]).layerOutputActivation[gradientId];
		grad = grad + actDelta*activationDerivative(act);
	}
	gradient->layerOutputBias[gradientId] = grad;
}

__kernel void updateNNparams(__global float* weights, __global float* gradient)
{
	int gradientId = get_global_id(0);
	float w = weights[gradientId];
	float g = gradient[gradientId];
	w = w - g*LEARNING_RATE;
	weights[gradientId] = w;
}

__kernel void cost(__global float* outputVector, __global int* outputIndex, __global Activation* activation, __global float* returnCost)
{
	float c = 0.0;
	for(int si = 0; si < STOCHASTIC_SAMPLING_SIZE; si++)
	{
		int oi = outputIndex[si];
		for(int i = 0; i < OUTPUT_SIZE; i++)
		{
			float err = (activation[si]).layerOutputActivation[i] - outputVector[oi*OUTPUT_SIZE + i];
			c = c + err*err;
		}
	}
	returnCost[0] += c/(2.0);
}

__kernel void normalizeGradient(__global float* gradient, int numSamples)
{
	int gradientId = get_global_id(0);
	gradient[gradientId] = gradient[gradientId] / (float)numSamples;
}
