/* 
Convolutional Neural Network
Greg Smith 2017

-----------------------------
	C1 feature map:

	5*5*6 feature map weights
	6 biases

	float weights[150];
	vectorized order : row, column, feature map 

	float biases[6];
-----------------------------

-----------------------------
	S2 feature map:

	6 weights;
	6 biases;

	float weights[6];
	float biases[6];
-----------------------------

-----------------------------
	C3 convolution layer:

	10*6*25 weights
	16 biases.

	float weights[1500];
	vectorized order: kernel index convolved with S2 (i.e. c3 index), s2 index, kernel row, kernel column, 

	float biases[16];
-----------------------------

-----------------------------
	S4 subsampling layer

	16 weights;
	16 biases;

	float weights[16];
	float biases[16];
-----------------------------

-----------------------------
	C5 convolution layer

	fully connect each pixel in each feature map to 120 different nodes.

	16*5*5*120 weights;
	120 biases;

	float weights[48000]
	vectorized order: C5 index, s4 feature map index, S4 row index, S4 column index,

	float biases[120]
-----------------------------

-----------------------------
	F6 layer

	fully connected

	84*120 weights;
	184 biases;

	float weight[10080]
	vectorized order: F6 index, C5 index
	float biases[184]
-----------------------------

-----------------------------
	output

	rbf classifier

	10*84 'rbf centers';

	float centers[840];
	vectorized order: output index, F6 index.
-----------------------------      

units in layers up to F6, the activation function applied.
xi = f(ai);
f(a) = Atanh(Sa); A = 1.7159;

*/

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.02
#endif

#ifndef STOCHASTIC_SAMPLING_SIZE
#define STOCHASTIC_SAMPLING_SIZE 40
#endif

#define BIAS_GRADIENT_SCALE 0.01

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5

#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 28
#endif

#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 28
#endif

#define INPUT_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT)

#define FEATURE_MAP_C_1_WIDTH (IMAGE_WIDTH - KERNEL_WIDTH + 1) 
#define FEATURE_MAP_C_1_HEIGHT (IMAGE_HEIGHT - KERNEL_HEIGHT + 1)
#define NUM_FEATURE_MAPS_C_1 6

#define FEATURE_MAP_S_2_WIDTH (FEATURE_MAP_C_1_WIDTH/2)
#define FEATURE_MAP_S_2_HEIGHT (FEATURE_MAP_C_1_HEIGHT/2)
#define NUM_FEATURE_MAPS_S_2 6

#define FEATURE_MAP_C_3_WIDTH (FEATURE_MAP_S_2_WIDTH - KERNEL_WIDTH + 1)
#define FEATURE_MAP_C_3_HEIGHT (FEATURE_MAP_S_2_HEIGHT - KERNEL_HEIGHT + 1)
#define NUM_FEATURE_MAPS_C_3 16

#define FEATURE_MAP_S_4_WIDTH (FEATURE_MAP_C_3_WIDTH/2)
#define FEATURE_MAP_S_4_HEIGHT (FEATURE_MAP_C_3_HEIGHT/2)
#define NUM_FEATURE_MAPS_S_4 16

#define NUM_FEATURE_MAPS_C_5 120

#define NUM_FEATURE_MAPS_F_6 84

#ifndef NUM_FEATURE_MAPS_OUTPUT
#define NUM_FEATURE_MAPS_OUTPUT 10
#endif

#define OUTPUT_SIZE NUM_FEATURE_MAPS_OUTPUT




#define C_1_WEIGHT_SIZE (KERNEL_WIDTH*KERNEL_HEIGHT*NUM_FEATURE_MAPS_C_1)
#define C_1_BIAS_SIZE NUM_FEATURE_MAPS_C_1

#define S_2_WEIGHT_SIZE NUM_FEATURE_MAPS_S_2
#define S_2_BIAS_SIZE NUM_FEATURE_MAPS_S_2

#define C_3_KERNELS_PER_S2 10
#define C_3_WEIGHT_SIZE KERNEL_WIDTH*KERNEL_HEIGHT*60
#define C_3_BIAS_SIZE NUM_FEATURE_MAPS_C_3


#define S_4_WEIGHT_SIZE NUM_FEATURE_MAPS_S_4
#define S_4_BIAS_SIZE NUM_FEATURE_MAPS_S_4

#define C_5_WEIGHT_SIZE FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4*NUM_FEATURE_MAPS_C_5
#define C_5_BIAS_SIZE NUM_FEATURE_MAPS_C_5

#define F_6_WEIGHT_SIZE NUM_FEATURE_MAPS_C_5*NUM_FEATURE_MAPS_F_6
#define F_6_BIAS_SIZE NUM_FEATURE_MAPS_F_6

#define OUTPUT_RBF_CENTERS NUM_FEATURE_MAPS_F_6*NUM_FEATURE_MAPS_OUTPUT


#define IMAGE_SIZE_INPUT IMAGE_WIDTH*IMAGE_HEIGHT
#define ACTIVATION_SIZE_C_1 (NUM_FEATURE_MAPS_C_1*FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT)
#define ACTIVATION_SIZE_S_2 (NUM_FEATURE_MAPS_S_2*FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT)
#define ACTIVATION_SIZE_C_3 (NUM_FEATURE_MAPS_C_3*FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT)
#define ACTIVATION_SIZE_S_4 (NUM_FEATURE_MAPS_S_4*FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT)
#define ACTIVATION_SIZE_C_5 (NUM_FEATURE_MAPS_C_5)
#define ACTIVATION_SIZE_F_6 (NUM_FEATURE_MAPS_F_6)
#define ACTIVATION_SIZE_OUTPUT (NUM_FEATURE_MAPS_OUTPUT)

#define ACTIVATION_BUFFER_SIZE (ACTIVATION_SIZE_C_1 + ACTIVATION_SIZE_S_2 + ACTIVATION_SIZE_C_3 + ACTIVATION_SIZE_S_4 + ACTIVATION_SIZE_C_5 + ACTIVATION_SIZE_F_6 + ACTIVATION_SIZE_OUTPUT)

typedef struct __Activations
{
	float activationC1[ACTIVATION_SIZE_C_1];
	float activationS2[ACTIVATION_SIZE_S_2];
	float activationC3[ACTIVATION_SIZE_C_3];
	float activationS4[ACTIVATION_SIZE_S_4];
	float activationC5[ACTIVATION_SIZE_C_5];
	float activationF6[ACTIVATION_SIZE_F_6];
	float activationOutput[ACTIVATION_SIZE_OUTPUT];
}Activations;


typedef struct __CNNparams
{
	float c1Weight[C_1_WEIGHT_SIZE];
	float c1Bias[C_1_BIAS_SIZE];

	float s2Weight[S_2_WEIGHT_SIZE];
	float s2Bias[S_2_BIAS_SIZE];

	float c3Weight[C_3_WEIGHT_SIZE];
	float c3Bias[C_3_BIAS_SIZE];

	float s4Weight[S_4_WEIGHT_SIZE];
	float s4Bias[S_4_BIAS_SIZE];

	float c5Weight[C_5_WEIGHT_SIZE];
	float c5Bias[C_5_BIAS_SIZE];

	float f6Weight[F_6_WEIGHT_SIZE];
	float f6Bias[F_6_BIAS_SIZE];

	float outputRBFCenter[OUTPUT_RBF_CENTERS];
} CNNparams;

__constant int s2c3Lookup[] = {0, -1, -1, -1, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, 8, 9,		
							10, 11, -1, -1, -1, 12, 13, 14, -1, -1, 15, 16, 17, 18, -1, 19,
							20, 21, 22, -1, -1, -1, 23, 24, 25, -1, -1, 26, -1, 27, 28, 29,
							-1, 30, 31, 32, -1, -1, 33, 34, 35, 36, -1, -1, 37, -1, 38, 39,
							-1, -1, 40, 41, 42, -1, -1, 43, 44, 45, 46, -1, 47, 48, -1, 49,
							-1, -1, -1, 50, 51, 52, -1, -1, 53, 54, 55, 56, -1, 57, 58, 59};

// HEADERS
__kernel void activationC1(__global float* input, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationS2(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationC3(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationS4(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationC5(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationF6(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationOutput(__global Activations* activations, __global CNNparams* cnnParams);


__kernel void activationDeltaC1(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);
__kernel void activationDeltaOutput(__global float* output, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta);

__kernel void addGradientWeightC1(__global float* input, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientWeightOutput(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);

__kernel void addGradientBiasC1(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);
__kernel void addGradientBiasOutput(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient);


__kernel void normalizeGradient(__global float* gradient, int numSamples);
__kernel void updateNNparams(__global float* weights, __global float* gradient);
__kernel void cost(__global float* outputVector, __global int* outputIndex, __global Activations* activation, __global float* returnCost);

/* 
=======================================================================================
========================  GET CNN PARAM FUNCTIONS =====================================
=======================================================================================
*/

float sampleImage(__global float* input, int imageIndex, int pixelIndex);
float sampleOutput(__global float* output, int outputIndex, int layerIndex);

float getC1Weight(__global CNNparams* cnnParams, int kernelXIndex, int kernelYIndex, int featureMapIndex);
float getC1Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getS2Weight(__global CNNparams* cnnParams, int featureMapIndex);
float getS2Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getC3Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC3Index, int featureMapS2Index);
float getC3Bias(__global CNNparams* cnnParams, int featureMapC3Index);

float getS4Weight(__global CNNparams* cnnParams, int featureMapIndex);
float getS4Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getC5Weight(__global CNNparams* cnnParams, int s4pixelX, int s4PixelY, int featureMapS4Index, int featureMapC5Index);
float getC5Bias(__global CNNparams* cnnParams, int featureMapC5Index);

float getF6Weight(__global CNNparams* cnnParams, int featureMapC5Index, int featureMapF6Index);
float getF6Bias(__global CNNparams* cnnParams, int featureMapF6Index);

float getOutputRBFCenterWeight(__global CNNparams* cnnParam, int featureMapF6Index, int featureMapOutputIndex);


float getActivationC1(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationS2(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationC3(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationS4(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationC5(__global Activations* activation, int featureMapIndex);
float getActivationF6(__global Activations* activation, int featureMapIndex);
float getActivationOutput(__global Activations* activation, int featureMapIndex);

void getActivationIndiciesC1(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesS2(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesC3(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesS4(int activationIndex, int* xIndex, int* yIndex, int* f);

void getWeightIndexC5(int activationIndex, int* s4IndexX, int* s4IndexY, int* s4featureMapIndex, int* c5Index);
void getWeightIndexF6(int activationIndex, int* c5Index, int* f6Index);
void getWeightIndexOutput(int activationIndex, int* f6Index, int* outputIndex);

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







































// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------
// ------------------------------------WEIGHT AND BIASES GETTERS ------------------------------------
// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

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

float getC1Weight(__global CNNparams* cnnParams, int kernelXIndex, int kernelYIndex, int featureMapIndex)
{
	int index = featureMapIndex*(KERNEL_WIDTH*KERNEL_HEIGHT);
	index += kernelYIndex*KERNEL_WIDTH;
	index += kernelXIndex;
	return cnnParams->c1Weight[index];
}

float getC1Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->c1Bias[featureMapIndex];
}


float getS2Weight(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s2Weight[featureMapIndex];
}

float getS2Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s2Bias[featureMapIndex];
}

float getC3Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC3Index, int featureMapS2Index)
{
	int index = s2c3Lookup[NUM_FEATURE_MAPS_C_3*featureMapS2Index + featureMapC3Index];
	index = index*(KERNEL_WIDTH*KERNEL_HEIGHT) + yIndex*KERNEL_WIDTH + xIndex;
	return cnnParams->c3Weight[index];
}

float getC3Bias(__global CNNparams* cnnParams, int featureMapC3Index)
{
	return cnnParams->c3Bias[featureMapC3Index];
}

float getS4Weight(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s4Weight[featureMapIndex];
}

float getS4Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s4Bias[featureMapIndex];
}

float getC5Weight(__global CNNparams* cnnParams, int s4pixelX, int s4PixelY, int featureMapS4Index, int featureMapC5Index)
{
	int index = featureMapC5Index*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4);
	index += featureMapS4Index*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);
	index += s4PixelY*FEATURE_MAP_S_4_WIDTH;
	index += s4pixelX;
	return cnnParams->c5Weight[index];
}
float getC5Bias(__global CNNparams* cnnParams, int featureMapC5Index)
{
	return cnnParams->c5Bias[featureMapC5Index];
}

float getF6Weight(__global CNNparams* cnnParams, int featureMapC5Index, int featureMapF6Index)
{
	int index = featureMapF6Index*NUM_FEATURE_MAPS_C_5;
	index += featureMapC5Index;
	return cnnParams->f6Weight[index];
}
float getF6Bias(__global CNNparams* cnnParams, int featureMapF6Index)
{
	return cnnParams->f6Bias[featureMapF6Index];
}

float getOutputRBFCenterWeight(__global CNNparams* cnnParam, int featureMapF6Index, int featureMapOutputIndex)
{
	int index = featureMapOutputIndex*NUM_FEATURE_MAPS_F_6 + featureMapF6Index;
	return cnnParam->outputRBFCenter[index];
}







































/*
======================================================================================================
======================== GET ACTIVATION FUNCTIONS ====================================================
======================================================================================================
*/

float getActivationC1(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int index = featureMapIndex*(FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT);
	index += yIndex*FEATURE_MAP_C_1_WIDTH;
	index += xIndex;
	return activation->activationC1[index];
}

float getActivationS2(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int index = featureMapIndex*FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT;
	index += yIndex*FEATURE_MAP_S_2_WIDTH;
	index += xIndex;
	return activation->activationS2[index];
}

float getActivationC3(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int index = featureMapIndex*FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT;
	index += yIndex*FEATURE_MAP_C_3_WIDTH;
	index += xIndex;
	return activation->activationC3[index];
}

float getActivationS4(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int index = featureMapIndex*FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT;
	index += yIndex*FEATURE_MAP_S_4_WIDTH;
	index += xIndex;
	return activation->activationS4[index];
}

float getActivationC5(__global Activations* activation, int featureMapIndex)
{
	return activation->activationC5[featureMapIndex];
}

float getActivationF6(__global Activations* activation, int featureMapIndex)
{
	return activation->activationF6[featureMapIndex];
}

float getActivationOutput(__global Activations* activation, int featureMapIndex)
{
	return activation->activationOutput[featureMapIndex];
}


void getActivationIndiciesC1(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_C_1_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_C_1_WIDTH;
	*xIndex = tempIndex;
}

void getActivationIndiciesS2(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_S_2_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_S_2_WIDTH;
	*xIndex = tempIndex;
}
void getActivationIndiciesC3(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_C_3_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_C_3_WIDTH;
	*xIndex = tempIndex;
}
void getActivationIndiciesS4(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_S_4_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_S_4_WIDTH;
	*xIndex = tempIndex;
}

void getWeightIndexC5(int activationIndex, int* s4IndexX, int* s4IndexY, int* s4featureMapIndex, int* c5Index)
{
	//int index = featureMapC5Index*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4);
	//index += featureMapS4Index*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);
	//index += s4PixelY*FEATURE_MAP_S_4_WIDTH;
	//index += s4pixelX;

	int tempIndex = activationIndex;
	
	*c5Index = activationIndex / (FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4);
	tempIndex = tempIndex - (*c5Index)*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4);

	*s4featureMapIndex = tempIndex / (FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);
	tempIndex = tempIndex - (*s4featureMapIndex)*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);

	*s4IndexY = tempIndex / (FEATURE_MAP_S_4_WIDTH);
	tempIndex = tempIndex - (*s4IndexY)*(FEATURE_MAP_S_4_WIDTH);

	*s4IndexX = tempIndex;
}

void getWeightIndexF6(int activationIndex, int* c5Index, int* f6Index)
{
	//int index = featureMapF6Index*NUM_FEATURE_MAPS_C_5;
	int tempIndex = activationIndex;
	*f6Index = tempIndex / NUM_FEATURE_MAPS_C_5;
	tempIndex = tempIndex - (*f6Index)*NUM_FEATURE_MAPS_C_5;
	*c5Index = tempIndex;
}

void getWeightIndexOutput(int activationIndex, int* f6Index, int* outputIndex)
{
	//int index = featureMapOutputIndex*NUM_FEATURE_MAPS_F_6 + featureMapF6Index;
	int tempIndex = activationIndex;
	*outputIndex = tempIndex / NUM_FEATURE_MAPS_F_6;
	tempIndex = tempIndex - (*outputIndex)*NUM_FEATURE_MAPS_F_6;
	*f6Index = tempIndex;
}


float activationFunction(float x)
{
	float s = 1.0/(1.0 + exp(-x));
	return s;
}

float activationDerivative(float x)
{
	float s = x;
	s = s*(1.0 - s);
	return s;
}

float tanHyperbolicDerivative(float x)
{
	float tanhResult = x;
	return (1.0 - tanhResult*tanhResult);
}






























/*
======================================================================================================
======================== COMPUTE ACTIVATION FUNCTIONS ================================================
======================================================================================================
*/
__kernel void activationC1(__global float* input, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_C_1;
	int act_y = activationId / ACTIVATION_SIZE_C_1;

	int imageSampleCenterX;
	int imageSampleCenterY;
	int featureMapIndex;

	getActivationIndiciesC1(act_x, &imageSampleCenterX, &imageSampleCenterY, &featureMapIndex);
	int kernelHalfX = (KERNEL_WIDTH / 2);
	int kernelHalfY = (KERNEL_HEIGHT / 2);

	imageSampleCenterX = imageSampleCenterX + kernelHalfX;
	imageSampleCenterY = imageSampleCenterY + kernelHalfY;

	float activationSum = 0.0;
	int inputImageX;
	int inputImageY;
	for(int x = 0; x < KERNEL_WIDTH; x++ )
	{
		for(int y = 0; y < KERNEL_HEIGHT; y++)
		{
			inputImageX = imageSampleCenterX - kernelHalfX + x; 
			inputImageY = imageSampleCenterY - kernelHalfY + y;

			float ker = getC1Weight(cnnParams, x, y, featureMapIndex);
			int imSamplePixelIndex = inputImageY*IMAGE_WIDTH + inputImageX;
			float imSample = sampleImage(input, sampleIndex[act_y], imSamplePixelIndex);
			activationSum = activationSum + ker*imSample;
		}
	}
	

	activationSum = activationSum + getC1Bias(cnnParams,featureMapIndex);
	activationSum = tanh(activationSum);

	(activations[act_y]).activationC1[act_x] = activationSum;
}

__kernel void activationS2(__global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_S_2;
	int act_y = activationId / ACTIVATION_SIZE_S_2;

	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesS2(act_x, &xIndex, &yIndex, &featureMapIndex);

	int a1x1, a1x2, a1y1, a1y2;
	
	a1x1 = xIndex*2;
	a1x2 = a1x1 + 1;
	a1y1 = yIndex*2;
	a1y2 = a1y1 + 1;

	float activationSum = 0.0;
	activationSum  += getActivationC1(&(activations[act_y]), a1x1, a1y1, featureMapIndex);
	activationSum  += getActivationC1(&(activations[act_y]), a1x1, a1y2, featureMapIndex);
	activationSum  += getActivationC1(&(activations[act_y]), a1x2, a1y1, featureMapIndex);
	activationSum  += getActivationC1(&(activations[act_y]), a1x2, a1y2, featureMapIndex);
	activationSum = activationSum / 4.0; 

	activationSum = activationSum*getS2Weight(cnnParams, featureMapIndex);
	activationSum = activationSum + getS2Bias(cnnParams, featureMapIndex);
	activationSum = tanh(activationSum);

	(activations[act_y]).activationS2[act_x] = activationSum;
}

__kernel void activationC3(__global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_C_3;
	int act_y = activationId / ACTIVATION_SIZE_C_3;

	int imageSampleCenterX;
	int imageSampleCenterY;
	int featureMapC3;

	getActivationIndiciesC3(act_x, &imageSampleCenterX, &imageSampleCenterY, &featureMapC3);

	int kernelHalfX = (KERNEL_WIDTH / 2);
	int kernelHalfY = (KERNEL_HEIGHT / 2);

	imageSampleCenterX = imageSampleCenterX + kernelHalfX;
	imageSampleCenterY = imageSampleCenterY + kernelHalfY;

	float activationSum = 0.0;
	for(int s2FeatureMap = 0; s2FeatureMap < NUM_FEATURE_MAPS_S_2; s2FeatureMap++)
	{
		int connectionIndex = NUM_FEATURE_MAPS_C_3*s2FeatureMap + featureMapC3;
		int kernelIndex = s2c3Lookup[connectionIndex];

		if(kernelIndex >= 0)
		{
			for(int x = 0; x < KERNEL_WIDTH; x++ )
			{
				for(int y = 0; y < KERNEL_HEIGHT; y++)
				{
					int imageX = imageSampleCenterX - kernelHalfX + x;
					int imageY = imageSampleCenterY - kernelHalfY + y;

					//float ker = getC3Weight(cnnParams, x, y, featureMapC3, s2FeatureMap);
					float ker = getC3Weight(cnnParams, x, y, featureMapC3, s2FeatureMap);
					float imSample = getActivationS2(&(activations[act_y]), imageX, imageY, s2FeatureMap);
					activationSum = activationSum + ker*imSample;
				}
			}
		}
	}	

	activationSum = activationSum + getC3Bias(cnnParams,featureMapC3);
	activationSum = tanh(activationSum);

	(activations[act_y]).activationC3[act_x] = activationSum;
}

__kernel void activationS4(__global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_S_4;
	int act_y = activationId / ACTIVATION_SIZE_S_4;

	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesS4(act_x, &xIndex, &yIndex, &featureMapIndex);

	int a1x1, a1x2, a1y1, a1y2;
	
	a1x1 = xIndex*2;
	a1x2 = a1x1 + 1;
	a1y1 = yIndex*2;
	a1y2 = a1y1 + 1;

	float activationSum = 0.0;
	activationSum  += getActivationC3(&(activations[act_y]), a1x1, a1y1, featureMapIndex);
	activationSum  += getActivationC3(&(activations[act_y]), a1x1, a1y2, featureMapIndex);
	activationSum  += getActivationC3(&(activations[act_y]), a1x2, a1y1, featureMapIndex);
	activationSum  += getActivationC3(&(activations[act_y]), a1x2, a1y2, featureMapIndex);
	activationSum = activationSum / 4.0; 

	activationSum = activationSum*getS4Weight(cnnParams, featureMapIndex);
	activationSum = activationSum + getS4Bias(cnnParams, featureMapIndex);
	activationSum = tanh(activationSum);
	(activations[act_y]).activationS4[act_x] = activationSum;
}

__kernel void activationC5(__global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_C_5;
	int act_y = activationId / ACTIVATION_SIZE_C_5;

	float activationSum = 0.0;
	for(int i = 0; i < ACTIVATION_SIZE_S_4; i++)
	{
		activationSum += (activations[act_y]).activationS4[i]*cnnParams->c5Weight[act_x*ACTIVATION_SIZE_S_4 + i];
	}
	
	activationSum = activationSum + getC5Bias(cnnParams,act_x);
	activationSum = tanh(activationSum);
	(activations[act_y]).activationC5[act_x] = activationSum;
}

__kernel void activationF6(__global Activations* activations, __global CNNparams* cnnParams)
{
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_F_6;
	int act_y = activationId / ACTIVATION_SIZE_F_6;

	float activationSum = 0.0;
	for(int i=0; i < ACTIVATION_SIZE_C_5; i++)
	{
		activationSum += (activations[act_y]).activationC5[i]*cnnParams->f6Weight[act_x*ACTIVATION_SIZE_C_5 + i];
	}

	activationSum += getF6Bias(cnnParams,act_x);
	activationSum = tanh(activationSum);
	(activations[act_y]).activationF6[act_x] = activationSum;
}

__kernel void activationOutput(__global Activations* activations, __global CNNparams* cnnParams)
{
	//10 centers with of 84 dimensions.
	int activationId = get_global_id(0);
	int act_x = activationId % ACTIVATION_SIZE_OUTPUT;
	int act_y = activationId / ACTIVATION_SIZE_OUTPUT;

	float activationSum = 0.0;
	for(int i=0 ; i < NUM_FEATURE_MAPS_F_6; i++)
	{
		float actf6 = (activations[act_y]).activationF6[i];
		float w = getOutputRBFCenterWeight(cnnParams, i, act_x);
		activationSum = activationSum + actf6*w;
	}
	activationSum = tanh(activationSum);
	(activations[act_y]).activationOutput[act_x] = activationSum;
}








/*
=========================================================================================================================
=============================================== COMPUTE ACTIVATION DELTAS ===============================================
=========================================================================================================================
*/
__kernel void activationDeltaC1(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)
{
	int c1Index = get_global_id(0);
	int c1_x = c1Index % ACTIVATION_SIZE_C_1;
	int c1_y = c1Index / ACTIVATION_SIZE_C_1;

	// get activation c3 indicies/
	float delta = 0.0;
	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesC1(c1_x, &xIndex, &yIndex, &featureMapIndex);

	// get activation s4 indicies
	int s2pixelX = xIndex / 2;
	int s2pixelY = yIndex / 2;

	// get the c1 s2 weight
	float w = getS2Weight(cnnParams, featureMapIndex);

	// compute the derivative
	delta = w / 4.0;
	delta = delta*getActivationS2(&(activationDelta[c1_y]), s2pixelX, s2pixelY, featureMapIndex);

	// multiply by the activation derivative
	float activationSample = getActivationS2(&(activations[c1_y]), s2pixelX, s2pixelY, featureMapIndex);
	delta = delta*tanHyperbolicDerivative(activationSample);

	(activationDelta[c1_y]).activationC1[c1_x] = delta;
}
__kernel void activationDeltaS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)
{
	// get activation index at s2
	int s2Index = get_global_id(0);
	int s2_x = s2Index % ACTIVATION_SIZE_S_2;
	int s2_y = s2Index / ACTIVATION_SIZE_S_2;

	int s2BaseIndexX;
	int s2BaseIndexY;
	int s2FeatureMap;
	getActivationIndiciesS2(s2_x, &s2BaseIndexX, &s2BaseIndexY, &s2FeatureMap);

	int kernelMidx = KERNEL_WIDTH/2;
	int kernelMidy = KERNEL_HEIGHT/2;
	int c3IndexX;
	int c3IndexY;
	int s2IndexX;
	int s2IndexY;

	float activationSample;
	float activationDeltaSample;
	float weightSample;
	float delta = 0.0;

	int numSamp = 0;

	for(int c3FeatureMap = 0; c3FeatureMap < NUM_FEATURE_MAPS_C_3; c3FeatureMap++)
	{
		int kernelIndex = s2c3Lookup[s2FeatureMap*NUM_FEATURE_MAPS_C_3 + c3FeatureMap];

		if(kernelIndex != 0)
		{
			for(int kernelX = 0; kernelX < KERNEL_WIDTH; kernelX++)
			{
				for(int kernelY = 0; kernelY < KERNEL_HEIGHT; kernelY++)
				{
					int sampleC3X = s2BaseIndexX - kernelX;
					int sampleC3Y = s2BaseIndexY - kernelY;

					if(sampleC3X >= 0 && sampleC3Y < FEATURE_MAP_C_3_WIDTH && sampleC3Y >= 0 && sampleC3Y < FEATURE_MAP_C_3_HEIGHT)
					{
						activationDeltaSample = getActivationC3(&(activations[s2_y]), sampleC3X, sampleC3Y, c3FeatureMap);
						activationSample = getActivationC3(&(activationDelta[s2_y]), sampleC3X, sampleC3Y, c3FeatureMap);
						weightSample = getC3Weight(cnnParams, kernelX, kernelY, c3FeatureMap, s2FeatureMap);

						delta = delta + activationDeltaSample*tanHyperbolicDerivative(activationSample)*weightSample;
						numSamp++;
					}
				}
			}
		}
	}


	(activationDelta[s2_y]).activationS2[s2_x] = delta/(float)numSamp;
}
__kernel void activationDeltaC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)
{
	int c3Index = get_global_id(0);
	int c3_x = c3Index % ACTIVATION_SIZE_C_3;
	int c3_y = c3Index / ACTIVATION_SIZE_C_3;

	// get activation c3 indicies/
	float delta = 0.0;
	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesC3(c3_x, &xIndex, &yIndex, &featureMapIndex);

	// get activation s4 indicies
	int s4pixelX = xIndex / 2;
	int s4pixelY = yIndex / 2;

	// get the c3 s4 weight
	float w = getS4Weight(cnnParams, featureMapIndex);
	
	// compute the derivative
	delta = w / 4.0;
	delta = delta*getActivationS4(&(activationDelta[c3_y]), s4pixelX, s4pixelY, featureMapIndex);

	// multiply by the activation derivative
	float activationSample = getActivationS4(&(activations[c3_y]), s4pixelX, s4pixelY, featureMapIndex);
	delta = delta*tanHyperbolicDerivative(activationSample);

	(activationDelta[c3_y]).activationC3[c3_x] = delta;
}
__kernel void activationDeltaS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)
{
	int s4Index = get_global_id(0);
	int s4_x = s4Index % ACTIVATION_SIZE_S_4;
	int s4_y = s4Index / ACTIVATION_SIZE_S_4;

	float delta = 0.0;
	float activationC5Sample = 0.0;
	float activationC5DeltaSample = 0.0;
	float weightSample = 0.0;

	int s4PixelX;
	int s4PixelY;
	int featureMapIndex;

	getActivationIndiciesS4(s4_x, &s4PixelX, &s4PixelY, &featureMapIndex);

	for(int c5Index = 0; c5Index < ACTIVATION_SIZE_C_5; c5Index++)
	{
		activationC5DeltaSample = (activationDelta[s4_y]).activationC5[c5Index];
		activationC5Sample = (activations[s4_y]).activationC5[c5Index];

		weightSample = getC5Weight(cnnParams, s4PixelX, s4PixelY, featureMapIndex, c5Index);
		delta += activationC5DeltaSample*weightSample*tanHyperbolicDerivative(activationC5Sample);
	}
	(activationDelta[s4_y]).activationS4[s4_x] = delta;
}
__kernel void activationDeltaC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)	
{
	int c5ActivationIndex = get_global_id(0);
	int c5_x = c5ActivationIndex % ACTIVATION_SIZE_C_5;
	int c5_y = c5ActivationIndex / ACTIVATION_SIZE_C_5;

	float delta = 0.0;
	float activationDeltaSample = 0.0;
	float weightSample = 0.0;

	for(int f6Index = 0; f6Index < ACTIVATION_SIZE_F_6; f6Index++)
	{
		activationDeltaSample = (activationDelta[c5_y]).activationF6[f6Index];
		weightSample = getF6Weight(cnnParams, c5_x, f6Index);
		delta += activationDeltaSample*weightSample;
	}
	(activationDelta[c5_y]).activationC5[c5_x] = delta;
}
__kernel void activationDeltaF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)	
{
	int f6Index = get_global_id(0);
	int f6_x = f6Index % ACTIVATION_SIZE_F_6;
	int f6_y = f6Index / ACTIVATION_SIZE_F_6;

	float delta = 0.0;
	float activationSample;
	float activationSampleDelta;
	float rbfSample;
	float d;
	float activationf6 = getActivationF6(&(activations[f6_y]),f6_x);

	for(int outputIndex = 0; outputIndex < NUM_FEATURE_MAPS_OUTPUT; outputIndex++)
	{
		activationSample = getActivationOutput(&(activations[f6_y]), outputIndex);
		activationSampleDelta = getActivationOutput(&(activationDelta[f6_y]),outputIndex);
		rbfSample = getOutputRBFCenterWeight(cnnParams, f6_x, outputIndex);

		d = tanHyperbolicDerivative(activationSample)*activationSampleDelta*rbfSample;
		delta = delta + d;
	}
	(activationDelta[f6_y]).activationF6[f6_x] = delta;
}
__kernel void activationDeltaOutput(__global float* output, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationDelta)
{
	int activationIndex = get_global_id(0);
	int act_x = activationIndex % ACTIVATION_SIZE_OUTPUT;
	int act_y = activationIndex / ACTIVATION_SIZE_OUTPUT;

	float delta = (activations[act_y].activationOutput[act_x] - sampleOutput(output, sampleIndex[act_y], act_x));
	(activationDelta[act_y]).activationOutput[act_x] = delta;
}



/*
===========================================================================================
============================ COMPUTE GRADIENTS ============================================
===========================================================================================
*/

__kernel void addGradientWeightC1(__global float* input, __global int* sampleIndex, __global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int weightIndex = get_global_id(0);

	int c1FeatureMapIndex = weightIndex / (KERNEL_WIDTH*KERNEL_HEIGHT);
	int tempIndex = weightIndex - c1FeatureMapIndex*(KERNEL_WIDTH*KERNEL_HEIGHT);
	int kernelY = tempIndex / KERNEL_WIDTH;
	tempIndex = tempIndex - kernelY*KERNEL_WIDTH;
	int kernelX = tempIndex;

	float activationC1Sample;
	float activationC1DeltaSample;
	float inputSample;
	float grad = 0.0;
	float numSamples = 0.0;

	for(int i = 0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x = 0; x < FEATURE_MAP_C_1_WIDTH; x++)
		{
			for(int y =0 ; y < FEATURE_MAP_C_1_HEIGHT; y++)
			{
				int inputX = (x + kernelX);
				int inputY = (y + kernelY);

				activationC1Sample = getActivationC1(&(activations[i]), x, y, c1FeatureMapIndex);
				activationC1DeltaSample = getActivationC1(&(activationsDelta[i]), x, y, c1FeatureMapIndex);
				int imagePixelIndex = inputY*IMAGE_WIDTH + inputX;
				inputSample = sampleImage(input, sampleIndex[i], imagePixelIndex);

				grad = grad + activationC1DeltaSample*tanHyperbolicDerivative(activationC1Sample)*inputSample;
				numSamples = numSamples + 1.0;
			}
		}
	}
	gradient->c1Weight[weightIndex] = gradient->c1Weight[weightIndex] + grad/numSamples;
}

__kernel void addGradientWeightS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int s2FeatureMapIndex = get_global_id(0);

	float activationS2Sample;
	float activationS2DeltaSample;
	float actC1x1y1Sample;
	float actC1x1y2Sample;
	float actC1x2y1Sample;
	float actC1x2y2Sample;

	float grad = 0.0;
	float numSamples = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x = 0; x < FEATURE_MAP_S_2_WIDTH; x++)
		{
			for(int y = 0; y < FEATURE_MAP_S_2_HEIGHT; y++)
			{
				activationS2Sample = getActivationS2(&(activations[i]), x, y, s2FeatureMapIndex);
				activationS2DeltaSample = getActivationS2(&(activationsDelta[i]), x, y, s2FeatureMapIndex);

				int c1x = x/2;
				int c1y = y/2;

				actC1x1y1Sample = getActivationC1(&(activations[i]), c1x, c1y, s2FeatureMapIndex);
				actC1x1y2Sample = getActivationC1(&(activations[i]), c1x, c1y + 1, s2FeatureMapIndex);
				actC1x2y1Sample = getActivationC1(&(activations[i]), c1x + 1, c1y, s2FeatureMapIndex);
				actC1x2y2Sample = getActivationC1(&(activations[i]), c1x + 1, c1y + 1, s2FeatureMapIndex);
				float c1Avg = (actC1x1y1Sample + actC1x1y2Sample + actC1x2y1Sample + actC1x2y2Sample)*0.25;

				grad = grad + activationS2DeltaSample*tanHyperbolicDerivative(activationS2Sample)*c1Avg;
				numSamples = numSamples + 1.0;
			}
		}
	}

	gradient->s2Weight[s2FeatureMapIndex] = gradient->s2Weight[s2FeatureMapIndex] + grad/numSamples;
}

__kernel void addGradientWeightC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int weightIndex = get_global_id(0);

	int kernelIndex = weightIndex / (KERNEL_WIDTH*KERNEL_HEIGHT);

	int c3FeatureMapIndex;
	int s2FeatureMapIndex;
	int fmIndex = 0;

	int kernelMidX = KERNEL_WIDTH / 2;
	int kernelMidy = KERNEL_HEIGHT / 2;
	int kernelX;
	int kernelY;

	for(s2FeatureMapIndex = 0; s2FeatureMapIndex < NUM_FEATURE_MAPS_S_2; s2FeatureMapIndex++)
	{
		for(c3FeatureMapIndex = 0; c3FeatureMapIndex < NUM_FEATURE_MAPS_C_3; c3FeatureMapIndex++)
		{
			if(s2c3Lookup[fmIndex] == kernelIndex)
				break;

			fmIndex++;
		}
	}

	int tempIndex = weightIndex - kernelIndex*(KERNEL_WIDTH*KERNEL_HEIGHT);
	kernelY = tempIndex / KERNEL_WIDTH;
	tempIndex = tempIndex - kernelY*KERNEL_WIDTH;
	kernelX = tempIndex;

	float activationS2Sample;
	float activationDeltaC3Sample;
	float activationC3Sample;
	float grad = 0.0;
	float numSamples = 0.0;

	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x = 0; x < FEATURE_MAP_C_3_WIDTH; x++)
		{
			for(int y =0 ; y < FEATURE_MAP_C_3_HEIGHT; y++)
			{
				int s2X = x + kernelX;
				int s2Y = y + kernelY;

				activationS2Sample = getActivationS2(&(activations[i]), s2X, s2Y, s2FeatureMapIndex);
				activationC3Sample = getActivationC3(&(activations[i]), x, y, c3FeatureMapIndex);
				activationDeltaC3Sample = getActivationC3(&(activationsDelta[i]), x, y, c3FeatureMapIndex);

				grad = grad + activationDeltaC3Sample*tanHyperbolicDerivative(activationC3Sample)*activationS2Sample;
				numSamples = numSamples + 1.0;
			}
		}
	}
	gradient->c3Weight[weightIndex] = gradient->c3Weight[weightIndex] + grad/numSamples;
}

__kernel void addGradientWeightS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int s4FeatureMapIndex = get_global_id(0);

	float activationS4Sample;
	float activations4DeltaSample;
	float actC3x1y1Sample;
	float actC3x1y2Sample;
	float actC3x2y1Sample;
	float actC3x2y2Sample;

	float grad = 0.0;
	float numSamples = 0.0;

	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x = 0; x < FEATURE_MAP_S_4_WIDTH; x++)
		{
			for(int y = 0; y < FEATURE_MAP_S_4_HEIGHT; y++)
			{
				activationS4Sample = getActivationS4(&(activations[i]), x, y, s4FeatureMapIndex);
				activations4DeltaSample = getActivationS4(&(activationsDelta[i]), x, y, s4FeatureMapIndex);

				int c3x = x/2;
				int c3y = y/2;

				actC3x1y1Sample = getActivationC3(&(activations[i]), c3x, c3y, s4FeatureMapIndex);
				actC3x1y2Sample = getActivationC3(&(activations[i]), c3x, c3y + 1, s4FeatureMapIndex);
				actC3x2y1Sample = getActivationC3(&(activations[i]), c3x + 1, c3y, s4FeatureMapIndex);
				actC3x2y2Sample = getActivationC3(&(activations[i]), c3x + 1, c3y + 1, s4FeatureMapIndex);
				float c3Avg = (actC3x1y1Sample + actC3x1y2Sample + actC3x2y1Sample + actC3x2y2Sample)*0.25;

				grad = grad + activations4DeltaSample*tanHyperbolicDerivative(activationS4Sample)*c3Avg;
				numSamples = numSamples + 1.0;
			}
		}
	}

	gradient->s4Weight[s4FeatureMapIndex] = gradient->s4Weight[s4FeatureMapIndex] + grad/numSamples;
}

__kernel void addGradientWeightC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int weightIndex = get_global_id(0);
	int s4IndexX;
	int s4IndexY;
	int s4featureMapIndex;
	int c5Index;
	float grad = 0.0;
	getWeightIndexC5(weightIndex, &s4IndexX, &s4IndexY, &s4featureMapIndex, &c5Index);

	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float activationS4Sample = getActivationS4(&(activations[i]), s4IndexX, s4IndexY, s4featureMapIndex);
		float activationC5Sample = getActivationC5(&(activations[i]), c5Index);
		float activationDeltaC5Sample = getActivationC5(&(activationsDelta[i]), c5Index);
		grad = grad + activationDeltaC5Sample*tanHyperbolicDerivative(activationC5Sample)*activationS4Sample;
	}
	gradient->c5Weight[weightIndex] = gradient->c5Weight[weightIndex] + grad/((float)STOCHASTIC_SAMPLING_SIZE);
}

__kernel void addGradientWeightF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int weightIndex = get_global_id(0);
	int c5Index;
	int f6Index;
	float grad = 0.0;
	getWeightIndexF6(weightIndex, &c5Index, &f6Index);

	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float activationF6DeltaSample = getActivationF6(&(activationsDelta[i]), f6Index);
		float activationC5Sample = getActivationC5(&(activations[i]),c5Index);
		grad = grad + activationF6DeltaSample*activationC5Sample;
	}
	gradient->f6Weight[weightIndex] = gradient->f6Weight[weightIndex] + grad/((float)STOCHASTIC_SAMPLING_SIZE);
}

__kernel void addGradientWeightOutput(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int weightIndex = get_global_id(0);

	//int index = featureMapOutputIndex*NUM_FEATURE_MAPS_F_6 + featureMapF6Index;
	int featureMapOutputIndex;
	int featureMapF6Index;
	getWeightIndexOutput(weightIndex, &featureMapF6Index, &featureMapOutputIndex);
	float grad = 0.0;

	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float activationF6Sample = getActivationF6(&(activations[i]), featureMapF6Index);
		float activationOutputDeltaSample = getActivationOutput(&(activationsDelta[i]), featureMapOutputIndex);
		float activationOutputSample = getActivationOutput(&(activations[i]),featureMapOutputIndex);
		float weightSample = getOutputRBFCenterWeight(cnnParams, featureMapF6Index, featureMapOutputIndex);
		grad = grad + activationOutputDeltaSample*tanHyperbolicDerivative(activationOutputSample)*activationF6Sample;
	}
	gradient->outputRBFCenter[weightIndex] = gradient->outputRBFCenter[weightIndex] + grad/((float)STOCHASTIC_SAMPLING_SIZE);
}



__kernel void addGradientBiasC1(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);

	float activationDeltaSample;
	float activationSample;
	float grad = 0.0;
	float numSamples = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x=0; x < FEATURE_MAP_C_1_WIDTH; x++)
		{
			for(int y = 0; y < FEATURE_MAP_C_1_HEIGHT; y++)
			{
				activationDeltaSample = getActivationC1(&(activationsDelta[i]), x, y, featureMapIndex);
				activationSample = getActivationC1(&(activations[i]), x, y, featureMapIndex);
				grad = grad + activationDeltaSample*tanHyperbolicDerivative(activationSample);
				numSamples = numSamples + 1.0;
			}
		}
	}
	gradient->c1Bias[featureMapIndex] = gradient->c1Bias[featureMapIndex] + BIAS_GRADIENT_SCALE*grad/numSamples;
}

__kernel void addGradientBiasS2(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);

	float activationDeltaSample;
	float activationSample;
	float grad = 0.0;
	float numSamples = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x=0; x < FEATURE_MAP_S_2_WIDTH; x++)
		{
			for(int y=0; y < FEATURE_MAP_S_2_HEIGHT; y++)
			{
				activationDeltaSample = getActivationS2(&(activationsDelta[i]), x, y, featureMapIndex);
				activationSample = getActivationS2(&(activations[i]), x, y, featureMapIndex);
				grad = grad + activationDeltaSample*tanHyperbolicDerivative(activationSample);
				numSamples = numSamples + 1.0;
			}
		}
	}
	gradient->s2Bias[featureMapIndex] = gradient->s2Bias[featureMapIndex] + BIAS_GRADIENT_SCALE*grad/numSamples;
}

__kernel void addGradientBiasC3(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);

	float activationDeltaSample;
	float activationSample;
	float grad = 0.0;
	float numSamples = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x=0; x < FEATURE_MAP_C_3_WIDTH; x++)
		{
			for(int y=0; y < FEATURE_MAP_C_3_HEIGHT; y++)
			{
				activationDeltaSample = getActivationC3(&(activationsDelta[i]), x, y, featureMapIndex);
				activationSample = getActivationC3(&(activations[i]), x, y, featureMapIndex);
				grad = grad + activationDeltaSample*tanHyperbolicDerivative(activationSample);
				numSamples = numSamples + 1.0;
			}
		}
	}
	gradient->c3Bias[featureMapIndex] = gradient->c3Bias[featureMapIndex] + BIAS_GRADIENT_SCALE*grad/numSamples;
}

__kernel void addGradientBiasS4(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);

	float activationDeltaSample;
	float activationSample;
	float grad = 0.0;
	float numSamples = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		for(int x=0; x < FEATURE_MAP_S_4_WIDTH; x++)
		{
			for(int y=0; y < FEATURE_MAP_S_4_HEIGHT; y++)
			{
				activationDeltaSample = getActivationS4(&(activationsDelta[i]), x, y, featureMapIndex);
				activationSample = getActivationS4(&(activations[i]), x, y, featureMapIndex);
				grad = grad + activationDeltaSample*tanHyperbolicDerivative(activationSample);
				numSamples = numSamples + 1.0;
			}
		}
	}

	gradient->s4Bias[featureMapIndex] = gradient->s4Bias[featureMapIndex] + BIAS_GRADIENT_SCALE*grad/numSamples;
}

__kernel void addGradientBiasC5(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);
	float grad = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float activationSample = getActivationC5(&(activations[i]), featureMapIndex);
		float activationDeltaSample = getActivationC5(&(activationsDelta[i]), featureMapIndex);
		grad = grad + activationDeltaSample*tanHyperbolicDerivative(activationSample);
	}
	gradient->c5Bias[featureMapIndex] = gradient->c5Bias[featureMapIndex] + BIAS_GRADIENT_SCALE*grad/((float)STOCHASTIC_SAMPLING_SIZE);
}

__kernel void addGradientBiasF6(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int f6Index = get_global_id(0);
	float grad = 0.0;
	for(int i=0; i < STOCHASTIC_SAMPLING_SIZE; i++)
	{
		float activationDeltaSample = getActivationF6(activationsDelta, f6Index);
		grad = grad + activationDeltaSample;
	}
	gradient->f6Bias[f6Index] = gradient->f6Bias[f6Index] + BIAS_GRADIENT_SCALE*grad/((float)STOCHASTIC_SAMPLING_SIZE);
}

__kernel void addGradientBiasOutput(__global Activations* activations, __global CNNparams* cnnParams, __global Activations* activationsDelta, __global CNNparams* gradient)
{
	int featureMapIndex = get_global_id(0);
	gradient->outputRBFCenter[featureMapIndex] = 0.0;
}

__kernel void normalizeGradient(__global float* gradient, int numSamples)
{
	int gradientId = get_global_id(0);
	gradient[gradientId] = gradient[gradientId] / (float)numSamples;
}
__kernel void updateNNparams(__global float* weights, __global float* gradient)
{
	int gradientId = get_global_id(0);
	float w = weights[gradientId];
	float g = gradient[gradientId];
	w = w - g*LEARNING_RATE;
	weights[gradientId] = w;
}
__kernel void cost(__global float* outputVector, __global int* outputIndex, __global Activations* activation, __global float* returnCost)
{
	float c = 0.0;
	for(int si = 0; si < STOCHASTIC_SAMPLING_SIZE; si++)
	{
		for(int i = 0; i < NUM_FEATURE_MAPS_OUTPUT; i++)
		{
			float err = (activation[si]).activationOutput[i] - outputVector[outputIndex[si]*OUTPUT_SIZE + i];
			c = c + err*err;
		}
	}
	returnCost[0] = returnCost[0] + c/2.0;
}