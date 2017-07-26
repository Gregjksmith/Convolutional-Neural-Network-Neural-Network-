/*
Fully Connected Neural Network, Convolutional Neural Network
Greg Smith
2017

*/

#pragma once


#include <stdlib.h>
#include <string>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <vector>
#include "CL\cl.h"

#include "ClContext.h"

#define NN_DEBUG 1

#define NN_KENRNEL_SOURCE "kernels/NeuralNetwork_v2.cl"

#define NN_INPUT_SIZE __inputSize
#define NN_LAYER_1_SIZE __activationLayer1Size
#define NN_LAYER_2_SIZE __activationLayer2Size
#define NN_OUTPUT_SIZE __activationOutputSize
#define NN_ACTIVATION_SIZE (NN_LAYER_1_SIZE + NN_LAYER_2_SIZE + NN_OUTPUT_SIZE)

#define STOCHASTIC_SAMPLING_SIZE __minibatchSize
#define SAMPLING_ITERATIONS __epochs

#define NN_LAYER_1_WEIGHT_SIZE		(NN_INPUT_SIZE*NN_LAYER_1_SIZE)
#define NN_LAYER_1_BIAS_SIZE	NN_LAYER_1_SIZE
#define NN_LAYER_2_WEIGHT_SIZE		(NN_LAYER_1_SIZE*NN_LAYER_2_SIZE)
#define NN_LAYER_2_BIAS_SIZE	NN_LAYER_2_SIZE
#define NN_LAYER_OUTPUT_WEIGHT_SIZE		(NN_LAYER_2_SIZE*NN_OUTPUT_SIZE)
#define NN_LAYER_OUTPUT_BIAS_SIZE		NN_OUTPUT_SIZE

#define NN_WEIGHT_SIZE (NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE + NN_LAYER_OUTPUT_WEIGHT_SIZE + NN_LAYER_OUTPUT_BIAS_SIZE)



#define CNN_KENRNEL_SOURCE "kernels/LeNet5_v2.cl"

#define CNN_STOCHASTIC_SAMPLING_SIZE __minibatchSize
#define CNN_SAMPLING_ITERATIONS __epochs

#define CNN_KERNEL_WIDTH 5
#define CNN_KERNEL_HEIGHT 5

#define CNN_IMAGE_WIDTH __inputWidth
#define CNN_IMAGE_HEIGHT __inputHeight

#define CNN_FEATURE_MAP_C_1_WIDTH (CNN_IMAGE_WIDTH - CNN_KERNEL_WIDTH + 1) 
#define CNN_FEATURE_MAP_C_1_HEIGHT (CNN_IMAGE_HEIGHT - CNN_KERNEL_HEIGHT + 1)
#define CNN_NUM_FEATURE_MAPS_C_1 6
#define CNN_FEATURE_MAP_S_2_WIDTH (CNN_FEATURE_MAP_C_1_WIDTH/2)
#define CNN_FEATURE_MAP_S_2_HEIGHT (CNN_FEATURE_MAP_C_1_HEIGHT/2)
#define CNN_NUM_FEATURE_MAPS_S_2 6
#define CNN_FEATURE_MAP_C_3_WIDTH (CNN_FEATURE_MAP_S_2_WIDTH - CNN_KERNEL_WIDTH + 1)
#define CNN_FEATURE_MAP_C_3_HEIGHT (CNN_FEATURE_MAP_S_2_HEIGHT - CNN_KERNEL_HEIGHT + 1)
#define CNN_NUM_FEATURE_MAPS_C_3 16
#define CNN_FEATURE_MAP_S_4_WIDTH (CNN_FEATURE_MAP_C_3_WIDTH/2)
#define CNN_FEATURE_MAP_S_4_HEIGHT (CNN_FEATURE_MAP_C_3_HEIGHT/2)
#define CNN_NUM_FEATURE_MAPS_S_4 16
#define CNN_NUM_FEATURE_MAPS_C_5 120
#define CNN_NUM_FEATURE_MAPS_F_6 84
#define CNN_NUM_FEATURE_MAPS_OUTPUT __activationOutputSize


#define CNN_C_1_WEIGHT_SIZE (CNN_KERNEL_WIDTH*CNN_KERNEL_HEIGHT*CNN_NUM_FEATURE_MAPS_C_1)
#define CNN_C_1_BIAS_SIZE CNN_NUM_FEATURE_MAPS_C_1
#define CNN_S_2_WEIGHT_SIZE CNN_NUM_FEATURE_MAPS_S_2
#define CNN_S_2_BIAS_SIZE CNN_NUM_FEATURE_MAPS_S_2
#define CNN_C_3_KERNELS_PER_S2 10
#define CNN_C_3_WEIGHT_SIZE CNN_KERNEL_WIDTH*CNN_KERNEL_HEIGHT*60
#define CNN_C_3_BIAS_SIZE CNN_NUM_FEATURE_MAPS_C_3
#define CNN_S_4_WEIGHT_SIZE CNN_NUM_FEATURE_MAPS_S_4
#define CNN_S_4_BIAS_SIZE CNN_NUM_FEATURE_MAPS_S_4
#define CNN_C_5_WEIGHT_SIZE CNN_FEATURE_MAP_S_4_WIDTH*CNN_FEATURE_MAP_S_4_HEIGHT*CNN_NUM_FEATURE_MAPS_S_4*CNN_NUM_FEATURE_MAPS_C_5
#define CNN_C_5_BIAS_SIZE CNN_NUM_FEATURE_MAPS_C_5
#define CNN_F_6_WEIGHT_SIZE CNN_NUM_FEATURE_MAPS_C_5*CNN_NUM_FEATURE_MAPS_F_6
#define CNN_F_6_BIAS_SIZE CNN_NUM_FEATURE_MAPS_F_6
#define CNN_OUTPUT_RBF_CENTERS CNN_NUM_FEATURE_MAPS_F_6*CNN_NUM_FEATURE_MAPS_OUTPUT

#define CNN_INPUT_SIZE CNN_IMAGE_WIDTH*CNN_IMAGE_HEIGHT

#define CNN_IMAGE_SIZE_INPUT CNN_IMAGE_WIDTH*CNN_IMAGE_HEIGHT
#define CNN_ACTIVATION_SIZE_C_1 CNN_NUM_FEATURE_MAPS_C_1*CNN_FEATURE_MAP_C_1_WIDTH*CNN_FEATURE_MAP_C_1_HEIGHT
#define CNN_ACTIVATION_SIZE_S_2 CNN_NUM_FEATURE_MAPS_S_2*CNN_FEATURE_MAP_S_2_WIDTH*CNN_FEATURE_MAP_S_2_HEIGHT
#define CNN_ACTIVATION_SIZE_C_3 CNN_NUM_FEATURE_MAPS_C_3*CNN_FEATURE_MAP_C_3_WIDTH*CNN_FEATURE_MAP_C_3_HEIGHT
#define CNN_ACTIVATION_SIZE_S_4 CNN_NUM_FEATURE_MAPS_S_4*CNN_FEATURE_MAP_S_4_WIDTH*CNN_FEATURE_MAP_S_4_HEIGHT
#define CNN_ACTIVATION_SIZE_C_5 CNN_NUM_FEATURE_MAPS_C_5
#define CNN_ACTIVATION_SIZE_F_6 CNN_NUM_FEATURE_MAPS_F_6
#define CNN_ACTIVATION_SIZE_OUTPUT CNN_NUM_FEATURE_MAPS_OUTPUT

#define CNN_ACTIVATION_BUFFER_SIZE (CNN_ACTIVATION_SIZE_C_1 + CNN_ACTIVATION_SIZE_S_2 + CNN_ACTIVATION_SIZE_C_3 + CNN_ACTIVATION_SIZE_S_4 + CNN_ACTIVATION_SIZE_C_5 + CNN_ACTIVATION_SIZE_F_6 + CNN_ACTIVATION_SIZE_OUTPUT)
#define CNN_PARAM_BUFER_SIZE (CNN_C_1_WEIGHT_SIZE + CNN_C_1_BIAS_SIZE + CNN_S_2_WEIGHT_SIZE + CNN_S_2_BIAS_SIZE + CNN_C_3_WEIGHT_SIZE + CNN_C_3_BIAS_SIZE + CNN_S_4_WEIGHT_SIZE + CNN_S_4_BIAS_SIZE + CNN_C_5_WEIGHT_SIZE + CNN_C_5_BIAS_SIZE + CNN_F_6_WEIGHT_SIZE + CNN_F_6_BIAS_SIZE + CNN_OUTPUT_RBF_CENTERS)




namespace gs
{
	class NeuralNetwork;
	class ClKernel;
	class ClContext;


	/*
	class CLContext
	Structure that contains handles for the openCL paltform, device, context and CommandQueue.
	Constructor will populate the fields automatically or throw an error.
	*/
	class ClContext
	{
	public:
		ClContext();
		virtual ~ClContext();

		cl_device_id deviceId;
		cl_platform_id platformId;
		cl_uint retNumDevices;
		cl_uint retNumPlatforms;
		cl_context context;
		cl_command_queue commandQueue;

	private:
		void createContext();
		void createQueue();

		void printPlatformInfo(cl_platform_id platformId, cl_uint retNumPlatforms);
		void printDeviceInfo(cl_device_id deviceId);
	};


	/*
	class ClKernel
	Base Kernel class. Contains kernel handles and functions used to enqueue kernels and add arguments.
	Loads the kernel source directory 'kernelSource' and builds the source.

	contructor :
	@param const char* kernel source.
	@param ClContext* context.
	*/
	class ClKernel
	{
	public:
		ClKernel(const char* kernelSource, ClContext* context, std::vector<std::string> &kernelDefines);
		virtual ~ClKernel();
		virtual void train();
		size_t totalWorkItems;
	protected:
		/*
		int createKernel
		creates a kernel object of 'kernelName'. 'kerneName' must be located in the source.

		@param const char* kernelName
		@return int, index of the kernel created.
		*/
		int createKernel(const char* kernelName);

		/*
		void addKernelArg
		adds an argument to the kernel in the kernel vector, indexed by kernelIndex.

		@param size_t kernelIndex: Index of the kernel in the kernel vector.
		@param int argId : argument index
		@param unsigned int bufferSize
		@param void* buffer.
		*/
		void addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer);

		/*
		void enqueueKernel
		enqueues and runs the specified kernel with 'totalWorkItems' work items.

		@param size_t kernelIndex.
		*/
		inline void enqueueKernel(size_t kernelIndex)
		{
			//compute the number of iterations, offset, global work items and local work items

			if (__kernel.size() <= kernelIndex)
			{
				printf("ERROR, kernel out of index. \n");
				return;
			}

			size_t returnSize;
			size_t maxWorkItemSize[3];
			maxWorkItemSize[0] = 0;
			maxWorkItemSize[1] = 0;
			maxWorkItemSize[2] = 0;
			clGetDeviceInfo(__context->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(cl_int), &maxWorkItemSize, &returnSize);

			__localWorkSize = 1;
			size_t workItemsPerIteration = maxWorkItemSize[0];
			size_t iterations = (totalWorkItems / workItemsPerIteration) + 1;
			size_t offset = 0;

			size_t* workItemIteration = new size_t[iterations];
			int i;
			for (i = 0; i < iterations - 1; i++)
			{
				workItemIteration[i] = workItemsPerIteration;
			}
			workItemIteration[i] = totalWorkItems % workItemsPerIteration;

			for (int i = 0; i < iterations; i++)
			{
				__globalWorkSize = workItemIteration[i];

				cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, 0, NULL, NULL);
				switch (ret)
				{
				case CL_INVALID_PROGRAM_EXECUTABLE:
					printf("ERROR, no successfully built program executable available for device. \n");
					break;

				case CL_INVALID_COMMAND_QUEUE:
					printf("ERROR, invalid command queue. \n");
					break;

				case CL_INVALID_KERNEL:
					printf("ERROR, invalid kernel. \n");
					break;

				case CL_INVALID_CONTEXT:
					printf("ERROR, invalid context. \n");
					break;

				case CL_INVALID_KERNEL_ARGS:
					printf("ERROR, the kernel arguments have not been specified. \n");
					break;

				case CL_INVALID_WORK_DIMENSION:
					printf("ERROR, invalid work dimensions. \n");
					break;

				case CL_INVALID_GLOBAL_WORK_SIZE:
					printf("ERROR, invalid work size. \n");
					break;

				case CL_INVALID_GLOBAL_OFFSET:
					printf("ERROR, invalid global offset. \n");
					break;

				case CL_INVALID_WORK_GROUP_SIZE:
					printf("ERROR, invalid work group size. \n");
					break;

				case CL_INVALID_WORK_ITEM_SIZE:
					printf("ERROR, invalid work item size. \n");
					break;

				case CL_MISALIGNED_SUB_BUFFER_OFFSET:
					printf("ERROR, misaligned sub buffer offset. \n");
					break;


				case CL_INVALID_IMAGE_SIZE:
					printf("ERROR, invalid image size. \n");
					break;

				case CL_OUT_OF_RESOURCES:
					printf("ERROR, out of resources. \n");
					break;

				case CL_MEM_OBJECT_ALLOCATION_FAILURE:
					printf("ERROR, memory object allocation failure. \n");
					break;

				case CL_INVALID_EVENT_WAIT_LIST:
					printf("ERROR, invalid wait list. \n");
					break;

				case CL_OUT_OF_HOST_MEMORY:
					printf("ERROR, out of host memory. \n");
					break;

				case CL_SUCCESS:
					//printf("Success, kernel enqueued. \n");
					break;
				}

				offset = offset + __globalWorkSize;

			}
			clFinish(__context->commandQueue);
			delete workItemIteration;
		}

		inline void enqueueKernel(size_t kernelIndex, size_t workItemSizeX, size_t workItemSizeY)
		{
			//compute the number of iterations, offset, global work items and local work items

			if (__kernel.size() <= kernelIndex)
			{
				printf("ERROR, kernel out of index. \n");
				return;
			}

			size_t returnSize;
			size_t maxWorkItemSize[3];
			maxWorkItemSize[0] = 0;
			maxWorkItemSize[1] = 0;
			maxWorkItemSize[2] = 0;
			clGetDeviceInfo(__context->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(cl_int), &maxWorkItemSize, &returnSize);

			totalWorkItems = workItemSizeX*workItemSizeY;

			__localWorkSize = 1;
			size_t workItemsPerIteration = maxWorkItemSize[0];

			size_t iterations = (totalWorkItems / workItemsPerIteration) + 1;
			size_t offset = 0;

			size_t workItemIteration[1024];
			int i;
			for (i = 0; i < iterations - 1; i++)
			{
				workItemIteration[i] = workItemsPerIteration;
			}
			workItemIteration[i] = totalWorkItems % workItemsPerIteration;

			for (int i = 0; i < iterations; i++)
			{
				__globalWorkSize = workItemIteration[i];

				cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, 0, NULL, NULL);
				switch (ret)
				{
				case CL_INVALID_PROGRAM_EXECUTABLE:
					printf("ERROR, no successfully built program executable available for device. \n");
					break;

				case CL_INVALID_COMMAND_QUEUE:
					printf("ERROR, invalid command queue. \n");
					break;

				case CL_INVALID_KERNEL:
					printf("ERROR, invalid kernel. \n");
					break;

				case CL_INVALID_CONTEXT:
					printf("ERROR, invalid context. \n");
					break;

				case CL_INVALID_KERNEL_ARGS:
					printf("ERROR, the kernel arguments have not been specified. \n");
					break;

				case CL_INVALID_WORK_DIMENSION:
					printf("ERROR, invalid work dimensions. \n");
					break;

				case CL_INVALID_GLOBAL_WORK_SIZE:
					printf("ERROR, invalid work size. \n");
					break;

				case CL_INVALID_GLOBAL_OFFSET:
					printf("ERROR, invalid global offset. \n");
					break;

				case CL_INVALID_WORK_GROUP_SIZE:
					printf("ERROR, invalid work group size. \n");
					break;

				case CL_INVALID_WORK_ITEM_SIZE:
					printf("ERROR, invalid work item size. \n");
					break;

				case CL_MISALIGNED_SUB_BUFFER_OFFSET:
					printf("ERROR, misaligned sub buffer offset. \n");
					break;


				case CL_INVALID_IMAGE_SIZE:
					printf("ERROR, invalid image size. \n");
					break;

				case CL_OUT_OF_RESOURCES:
					printf("ERROR, out of resources. \n");
					break;

				case CL_MEM_OBJECT_ALLOCATION_FAILURE:
					printf("ERROR, memory object allocation failure. \n");
					break;

				case CL_INVALID_EVENT_WAIT_LIST:
					printf("ERROR, invalid wait list. \n");
					break;

				case CL_OUT_OF_HOST_MEMORY:
					printf("ERROR, out of host memory. \n");
					break;

				case CL_SUCCESS:
					//printf("Success, kernel enqueued. \n");
					break;
				}

				offset = offset + __globalWorkSize;
			}
			clFinish(__context->commandQueue);
		}

		void createBuffer(cl_context context, cl_mem_flags memFlag, size_t bufferSize, cl_mem &memObj);

		cl_program __program;
		std::vector<cl_kernel> __kernel;
		ClContext* __context;
		std::vector<cl_mem> __memoryObjects;
	private:
		size_t __globalWorkSize;
		size_t __localWorkSize;

	};


	/*
	class NNKernel
	Neural Network class used for digit recognition.

	Contructor:
	@param CLContext*
	@param vector<Mat*> input image training set
	@param vector<unsigned char> image label training set.

	void train()
	trains the neural network parameters using the training set.

	unsigned char predict(Mat* image)
	predicts the label of the image using the neural network.
	@param Mat* input image

	@return redicted label.
	*/
	class NeuralNetwork : public ClKernel
	{
	public:
		/*CONSTRUCTOR*/
		NeuralNetwork(ClContext* context, size_t inputSize, size_t activationLayer1Size, size_t activationLayer2Size, size_t outputLayerSize, float learningRate, size_t minibatchSize, size_t epochs);
		/*DESTRUCTOR*/
		virtual ~NeuralNetwork();

		/*
		void train
		trains the Neural Network parameters using the training images and training labels.
		*/
		virtual void train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples);

		/*
		float totalCost
		computes the total cost to the neural network with the training set.
		*/
		float totalCost();

		/*
		unsigned char predict
		predicts the output given the input
		@param float* inputVector
		@param float* outputVector
		@param size_t numSamples
		*/
		void predict(float* inputVector, float* outputVector, size_t numSamples);

		void exportNNParams(char* filePath);
		void importNNParams(char* filePath);

#if NN_DEBUG
		void exportReport(char* filePath);
#endif
	
	protected:
		size_t __inputSize;
		size_t __activationLayer1Size;
		size_t __activationLayer2Size;
		size_t __activationOutputSize;
		float __learningRate;
		size_t __minibatchSize;
		size_t __epochs;

		/*BUFFERS AND VARIABLES*/
		float* __nnParams;
		float* __activations;
		float* __activationDeltas;
		float* __gradient;
		float __cost;

		float* __inputImageVector;
		float* __trainingLabelsVector;
		size_t __numTrainingSample;

	private:
		/*
		void addNNKernelArg
		adds kernel arguments to each kernel function
		*/
		void addNNKernelArg();

		/*
		void createBuffers
		creates all the buffers necessary for the Neural Network
		*/
		void createBuffers();

		/*
		void createBuffers
		creates the training dataset buffers necessary for the Neural Network
		@param float* inputImage : input training vector
		@param float* outputVector : output training vector
		@param size_t numTrainingSamples : number of training samples
		*/
		void createTrainingBuffers(float* inputImage, float* outputVector, size_t numTrainingSamples);

		/*
		void createBuffers
		creates an empty training dataset buffers
		*/
		void createTrainingBuffers();
		

		/*
		float* createImageVector
		vectorizes the image training set.
		@param vector<Mat*> : image training set

		@return float* : pointer to the images vector
		*/
		float* createImageVector(float* inputImage);

		/*
		float* createOutputVector
		vectorizes the output training labels.
		@param vector<unsigned char> : training labels

		@return float* : pointer to the labels vector
		*/
		float* createOutputVector(float* trainingLabels);

		/*
		void initNNParams
		initializes the neural network weights. The weights are sampled from a uniform distribution.
		*/
		void initNNParams();

		/*
		void initGradientVector
		sets the gradient memory objects to zero.
		*/
		void initGradientVector();

		/*
		void setImageIndex
		sets the image index of the relevant kernels
		*/
		inline void setImageIndex(int* index)
		{
			clEnqueueWriteBuffer(__context->commandQueue, __memobjTrainingIndex, CL_TRUE, 0, sizeof(int)*STOCHASTIC_SAMPLING_SIZE, index, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		/*functions that read memory object buffers into host memory*/
		inline void readNNParams()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjNNParamsVector, CL_TRUE, 0, NN_WEIGHT_SIZE * sizeof(float), (void*)__nnParams, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readActivations()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjActivationVector, CL_TRUE, 0, STOCHASTIC_SAMPLING_SIZE*NN_ACTIVATION_SIZE * sizeof(float), (void*)__activations, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readGradient()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE * sizeof(float), (void*)__gradient, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readActivationDelta()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjActivationDeltaVector, CL_TRUE, 0, STOCHASTIC_SAMPLING_SIZE*NN_ACTIVATION_SIZE * sizeof(float), (void*)__activationDeltas, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readCost()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjCost, CL_TRUE, 0, sizeof(float), (void*)&__cost, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readBuffers()
		{
			readNNParams();
			readActivations();
			readGradient();
			readActivationDelta();
			readCost();
		}

		/*
		void clearBuffers
		clears the gradient buffer.
		*/
		inline void clearGradient()
		{
			for (int i = 0; i < NN_WEIGHT_SIZE; i++)
			{
				__gradient[i] = 0.0;
			}
			clEnqueueWriteBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), __gradient, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		/*
		void clearCost
		clears the cost buffer.
		*/
		inline void clearCost()
		{
			__cost = 0;
			clEnqueueWriteBuffer(__context->commandQueue, __memobjCost, CL_TRUE, 0, sizeof(float), &__cost, 0, NULL, NULL);
		}

		/*
		void computeCost
		computes the NN cost from the activation buffer and the label buffer
		and adds it to the cost buffer.
		*/
		inline void computeCost()
		{
			enqueueKernel(13, 1, 1);
		}

		/*
		void normalizeGradient
		divides the gradient vector by the total number of training samples.
		(the cost is the average l2 norm)
		*/
		inline void normalizeGradient()
		{
			enqueueKernel(14, NN_WEIGHT_SIZE, 1);
		}

		/*
		void updateNNParams
		updates the NNweights of the Neural Network.
		*/
		inline void updateNNParams()
		{
			enqueueKernel(12, NN_WEIGHT_SIZE, 1);
		}

		/*
		float gradientInnerProduct
		computes the inner product of the gradient vector.
		*/
		inline float gradientInnerProduct()
		{
			float ip = 0.0;
			for (int i = 0; i < NN_WEIGHT_SIZE; i++)
			{
				ip += __gradient[i] * __gradient[i];
			}
			return ip;
		}

		/*
		void calculateActivationsLayer1
		computes the activations in layer 1 and places them in the activation buffer.
		*/
		inline void calculateActivationsLayer1()
		{
			enqueueKernel(0, NN_LAYER_1_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void calculateActivationsLayer2
		computes the activations in layer 2 and places them in the activation buffer.
		*/
		inline void calculateActivationsLayer2()
		{
			enqueueKernel(1, NN_LAYER_2_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void calculateActivationsLayerOutput
		computes the activations in the output layer and places them in the activation buffer.
		*/
		inline void calculateActivationsLayerOutput()
		{
			enqueueKernel(2, NN_OUTPUT_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void calculateActivationsDeltaLayer1
		computes the activations delta in layer 1 and places them in the activation delta buffer.
		*/
		inline void calculateActivationsDeltaLayer1()
		{
			enqueueKernel(3, NN_LAYER_1_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void calculateActivationsDeltaLayer2
		computes the activations delta in layer 2 and places them in the activation delta buffer.
		*/
		inline void calculateActivationsDeltaLayer2()
		{
			enqueueKernel(4, NN_LAYER_2_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void calculateActivationsDeltaLayerOutput
		computes the activations delta in the output layer and places them in the activation delta buffer.
		*/
		inline void calculateActivationsDeltaLayerOutput()
		{
			enqueueKernel(5, NN_OUTPUT_SIZE, STOCHASTIC_SAMPLING_SIZE);
		}

		/*
		void addGradientWeightLayer1
		computes the gradient of each weight in layer 1 and adds it to the gradient buffer
		*/
		inline void addGradientWeightLayer1()
		{
			enqueueKernel(6, NN_LAYER_1_WEIGHT_SIZE, 1);
		}

		/*
		void addGradientWeightLayer2
		computes the gradient of each weight in layer 2 and adds it to the gradient buffer
		*/
		inline void addGradientWeightLayer2()
		{
			enqueueKernel(7, NN_LAYER_2_WEIGHT_SIZE, 1);
		}


		/*
		void addGradientWeightLayerOutput
		computes the gradient of each weight in the output layer and adds it to the gradient buffer
		*/
		inline void addGradientWeightLayerOutput()
		{
			enqueueKernel(8, NN_LAYER_OUTPUT_WEIGHT_SIZE, 1);
		}

		/*
		void addGraidnetBiasLayer1
		computes the gradient of the biases in layer 1 and adds it to the gradient buffer.
		*/
		inline void addGradientBiasLayer1()
		{
			enqueueKernel(9, NN_LAYER_1_BIAS_SIZE, 1);
		}

		/*
		void addGraidnetBiasLayer2
		computes the gradient of the biases in layer 2 and adds it to the gradient buffer.
		*/
		inline void addGradientBiasLayer2()
		{
			enqueueKernel(10, NN_LAYER_2_BIAS_SIZE, 1);
		}

		/*
		void addGraidnetBiasLayerOutput
		computes the gradient of the biases in the output layer and adds it to the gradient buffer.
		*/
		inline void addGradientBiasLayerOutput()
		{
			enqueueKernel(11, NN_LAYER_OUTPUT_BIAS_SIZE, 1);
		}

		std::vector<std::string> kernelGlobals(int inputSize, int layer1Size, int layer2Size, int outputSize, float learningRate, int minibatchSize);

		/*MEMORY OBJECTS*/
		cl_mem __memobjInputVector;
		cl_mem __memobjOutputTruthVector;
		cl_mem __memobjActivationVector;
		cl_mem __memobjNNParamsVector;
		cl_mem __memobjActivationDeltaVector;
		cl_mem __memobjGradientVector;
		cl_mem __memobjCost;
		cl_mem __memobjLearningParameter;
		cl_mem __memobjTrainingIndex;

		/* REPORT VARIABLES*/
#if NN_DEBUG
		time_t __elapsedTime;
		std::vector<float> __miniBatchCostHistory;
#endif
		/*reused variables, avoids leaks*/
		float* __inputVector;
		int* __sampleIndex;
	};



	class ConvolutionalNeuralNetwork : public ClKernel
	{
	public:
		/*CONSTRUCTOR*/
		//CNNKernel(ClContext* context, std::vector<Mat*> &inputImage, std::vector<unsigned char> &trainingLabels);
		ConvolutionalNeuralNetwork(ClContext* context, size_t inputImageWidth, size_t inputImageHeight, size_t outputSize, float learningRate, size_t minibatchSize, size_t numEpochs);

		/*DESTRUCTOR*/
		virtual ~ConvolutionalNeuralNetwork();

		/*
		void train
		trains the Neural Network parameters using the training images and training labels.
		*/
		virtual void train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples);

		/*
		float totalCost
		computes the total cost to the neural network with the training set.
		*/
		float totalCost();

		/*
		unsigned char predict
		predicts the digit classification using the neural network.
		@return unsigned char, predicted digit [0,9]
		*/
		void predict(float* inputVector, float* outputVector, size_t numSamples);

		void exportNNParams(char* filePath);
		void importNNParams(char* filePath);
		void exportReport(char* filePath);

	private:
		/*BUFFERS AND VARIABLES*/
		float* __cnnParams;
		float* __activations;
		float* __activationDeltas;
		float* __gradient;
		float __cost;

		size_t __inputWidth;
		size_t __inputHeight;
		size_t __activationOutputSize;
		float __learningRate;
		size_t __minibatchSize;
		size_t __epochs;

		float* __trainingInputVector;
		float* __trainingOutputVector;
		size_t __numTrainingSample;

		void initNNParams();

		/*
		void createBuffers
		creates all the buffers necessary for the Neural Network
		@param vector<Mat*> : image training set.
		@param vector<unsigned char> : label training set.
		*/
		void createBuffers();

		void createTrainingBuffers(float* inputVector, float* outputVector, size_t numSamples);
		void createTrainingBuffers();

		/*
		void addNNKernelArg
		adds kernel arguments to each kernel function
		*/
		void addCNNKernelArg();

		/*
		float* createImageVector
		vectorizes the image training set.
		@param vector<Mat*> : image training set

		@return float* : pointer to the images vector
		*/
		float* createInputVector(float* inputVector, size_t numSamples);

		/*
		float* createOutputVector
		vectorizes the output training labels.
		@param vector<unsigned char> : training labels

		@return float* : pointer to the labels vector
		*/
		float* createOutputVector(float* outputVector, size_t numSamples);

		/*
		void initNNParams
		initializes the neural network weights. The weights are sampled from a uniform distribution.
		*/
		inline void updateNNParams()
		{
			enqueueKernel(29, CNN_PARAM_BUFER_SIZE, 1);
		}


		/*
		void initGradientVector
		sets the gradient memory objects to zero.
		*/
		inline void initGradientVector()
		{
			for (int i = 0; i < CNN_PARAM_BUFER_SIZE; i++)
			{
				__gradient[i] = 0.0;
			}
			cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, CNN_PARAM_BUFER_SIZE*sizeof(float), __gradient, 0, NULL, NULL);
		}


		/*
		void setImageIndex
		sets the image index of the relevant kernels
		*/
		inline void setImageIndex(int* index)
		{
			clEnqueueWriteBuffer(__context->commandQueue, __memobjTrainingIndex, CL_TRUE, 0, sizeof(int)*CNN_STOCHASTIC_SAMPLING_SIZE, index, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}


		/*functions that read memory object buffers into host memory*/
		inline void readCNNParams()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjCNNParamsVector, CL_TRUE, 0, CNN_PARAM_BUFER_SIZE * sizeof(float), (void*)__cnnParams, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readActivations()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjActivationVector, CL_TRUE, 0, CNN_STOCHASTIC_SAMPLING_SIZE*CNN_ACTIVATION_BUFFER_SIZE * sizeof(float), (void*)__activations, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readGradient()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, CNN_PARAM_BUFER_SIZE * sizeof(float), (void*)__gradient, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readActivationDelta()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjActivationDeltaVector, CL_TRUE, 0, CNN_STOCHASTIC_SAMPLING_SIZE*CNN_ACTIVATION_BUFFER_SIZE * sizeof(float), (void*)__activationDeltas, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readCost()
		{
			cl_int ret;
			ret = clEnqueueReadBuffer(__context->commandQueue, __memobjCost, CL_TRUE, 0, sizeof(float), (void*)&__cost, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}

		inline void readBuffers()
		{
			readCNNParams();
			readActivations();
			readGradient();
			readActivationDelta();
			readCost();
		}


		/*
		void clearBuffers
		clears the gradient buffer.
		*/
		inline void clearGradient()
		{
			for (int i = 0; i < CNN_PARAM_BUFER_SIZE; i++)
			{
				__gradient[i] = 0.0;
			}
			clEnqueueWriteBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, CNN_PARAM_BUFER_SIZE*sizeof(float), __gradient, 0, NULL, NULL);
			clFinish(__context->commandQueue);
		}



		/*
		void clearCost
		clears the cost buffer.
		*/
		inline void clearCost()
		{
			__cost = 0;
			clEnqueueWriteBuffer(__context->commandQueue, __memobjCost, CL_TRUE, 0, sizeof(float), &__cost, 0, NULL, NULL);
		}


		/*
		void computeCost
		computes the NN cost from the activation buffer and the label buffer
		and adds it to the cost buffer.
		*/
		inline void computeCost()
		{
			totalWorkItems = 1;
			enqueueKernel(30, 1, 1);
		}

		/*
		void normalizeGradient
		divides the gradient vector by the total number of training samples.
		(the cost is the average l2 norm)
		*/
		inline void normalizeGradient()
		{
			enqueueKernel(28, CNN_PARAM_BUFER_SIZE, 1);
		}


		/*
		float gradientInnerProduct
		computes the inner product of the gradient vector.
		*/
		inline float gradientInnerProduct()
		{
			float ip = 0.0;
			for (int i = 0; i < CNN_PARAM_BUFER_SIZE; i++)
			{
				ip += __gradient[i] * __gradient[i];
			}
			return ip;
		}


		inline void calculateActivationsC1()
		{
			enqueueKernel(0, CNN_ACTIVATION_SIZE_C_1, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsS2()
		{
			enqueueKernel(1, CNN_ACTIVATION_SIZE_S_2, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsC3()
		{
			enqueueKernel(2, CNN_ACTIVATION_SIZE_C_3, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsS4()
		{
			enqueueKernel(3, CNN_ACTIVATION_SIZE_S_4, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsC5()
		{
			enqueueKernel(4, CNN_ACTIVATION_SIZE_C_5, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsF6()
		{
			enqueueKernel(5, CNN_ACTIVATION_SIZE_F_6, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationsOutput()
		{
			enqueueKernel(6, CNN_ACTIVATION_SIZE_OUTPUT, CNN_STOCHASTIC_SAMPLING_SIZE);
		}


		inline void calculateActivationDeltaC1()
		{
			enqueueKernel(7, CNN_ACTIVATION_SIZE_C_1, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaS2()
		{
			enqueueKernel(8, CNN_ACTIVATION_SIZE_S_2, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaC3()
		{
			enqueueKernel(9, CNN_ACTIVATION_SIZE_C_3, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaS4()
		{
			enqueueKernel(10, CNN_ACTIVATION_SIZE_S_4, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaC5()
		{
			enqueueKernel(11, CNN_ACTIVATION_SIZE_C_5, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaF6()
		{
			enqueueKernel(12, CNN_ACTIVATION_SIZE_F_6, CNN_STOCHASTIC_SAMPLING_SIZE);
		}
		inline void calculateActivationDeltaOutput()
		{
			enqueueKernel(13, CNN_ACTIVATION_SIZE_OUTPUT, CNN_STOCHASTIC_SAMPLING_SIZE);
		}


		inline void addGradientWeightC1()
		{
			enqueueKernel(14, CNN_C_1_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightS2()
		{
			enqueueKernel(15, CNN_S_2_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightC3()
		{
			enqueueKernel(16, CNN_C_3_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightS4()
		{
			enqueueKernel(17, CNN_S_4_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightC5()
		{
			enqueueKernel(18, CNN_C_5_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightF6()
		{
			enqueueKernel(19, CNN_F_6_WEIGHT_SIZE, 1);
		}
		inline void addGradientWeightOutput()
		{
			enqueueKernel(20, CNN_OUTPUT_RBF_CENTERS, 1);
		}


		inline void addGradientBiasC1()
		{
			enqueueKernel(21, CNN_C_1_BIAS_SIZE, 1);
		}
		inline void addGradientBiasS2()
		{
			enqueueKernel(22, CNN_S_2_BIAS_SIZE, 1);
		}
		inline void addGradientBiasC3()
		{
			enqueueKernel(23, CNN_C_3_BIAS_SIZE, 1);
		}
		inline void addGradientBiasS4()
		{
			enqueueKernel(24, CNN_S_4_BIAS_SIZE, 1);
		}
		inline void addGradientBiasC5()
		{
			enqueueKernel(25, CNN_C_5_BIAS_SIZE, 1);
		}
		inline void addGradientBiasF6()
		{
			enqueueKernel(26, CNN_F_6_BIAS_SIZE, 1);
		}
		inline void addGradientBiasOutput()
		{
			totalWorkItems = 0;
			enqueueKernel(27, 0, 1);
		}

		std::vector<std::string> kernelGlobals(size_t inputWidth, size_t inputHeight, size_t outputSize, float learningRate, int minibatchSize);

		/*MEMORY OBJECTS*/
		cl_mem __memobjInputVector;
		cl_mem __memobjOutputTruthVector;
		cl_mem __memobjActivationVector;
		cl_mem __memobjCNNParamsVector;
		cl_mem __memobjActivationDeltaVector;
		cl_mem __memobjGradientVector;
		cl_mem __memobjCost;
		cl_mem __memobjLearningParameter;
		cl_mem __memobjTrainingIndex;

		/* REPORT VARIABLES*/
#if NN_DEBUG
		time_t __elapsedTime;
		std::vector<float>__miniBatchCostHistory;
#endif
		int* __sampleIndex;
	};


}