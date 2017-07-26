#include "../header/NeuralNetwork.h"


gs::NeuralNetwork::NeuralNetwork(ClContext* context, size_t inputSize, size_t activationLayer1Size, size_t activationLayer2Size, size_t outputLayerSize,
	float learningRate, size_t minibatchSize, size_t epochs) :
	ClKernel(NN_KENRNEL_SOURCE, context, kernelGlobals(inputSize, activationLayer1Size, activationLayer2Size, outputLayerSize, learningRate, minibatchSize))
{
	cl_int ret;

	createKernel("activationLayer1");
	createKernel("activationLayer2");
	createKernel("activationLayerOutput");

	createKernel("activationLayer1Delta");
	createKernel("activationLayer2Delta");
	createKernel("activationOutputDelta");

	createKernel("addGradientWeightLayer1");
	createKernel("addGradientWeightLayer2");
	createKernel("addGradientWeightLayerOutput");

	createKernel("addGradientBiasLayer1");
	createKernel("addGradientBiasLayer2");
	createKernel("addGradientBiasLayerOutput");

	createKernel("updateNNparams");
	createKernel("cost");
	createKernel("normalizeGradient");

	this->__inputSize = inputSize;
	this->__activationLayer1Size = activationLayer1Size;
	this->__activationLayer2Size = activationLayer2Size;
	this->__activationOutputSize = outputLayerSize;
	this->__learningRate = learningRate;

	this->__minibatchSize = minibatchSize;
	this->__epochs = epochs;
	this->__numTrainingSample = 0;

	__inputImageVector = nullptr;
	__trainingLabelsVector = nullptr;
	__nnParams = nullptr;
	__activations = nullptr;
	__activationDeltas = nullptr;
	__gradient = nullptr;

	__memobjInputVector = nullptr;
	__memobjOutputTruthVector = nullptr;
	__memobjActivationVector = nullptr;
	__memobjNNParamsVector = nullptr;
	__memobjActivationDeltaVector = nullptr;
	__memobjGradientVector = nullptr;
	__memobjCost = nullptr;
	__memobjTrainingIndex = nullptr;

	__inputVector = new float[NN_INPUT_SIZE];
	__sampleIndex = new int[STOCHASTIC_SAMPLING_SIZE];
}

gs::NeuralNetwork::~NeuralNetwork()
{
	clReleaseMemObject(__memobjInputVector);
	clReleaseMemObject(__memobjOutputTruthVector);
	clReleaseMemObject(__memobjActivationVector);
	clReleaseMemObject(__memobjNNParamsVector);
	clReleaseMemObject(__memobjActivationDeltaVector);
	clReleaseMemObject(__memobjGradientVector);
	clReleaseMemObject(__memobjCost);
	clReleaseMemObject(__memobjTrainingIndex);

	delete __inputImageVector;
	delete __trainingLabelsVector;

	delete __nnParams;
	delete __activations;
	delete __activationDeltas;
	delete __gradient;

	delete __inputVector;
	delete __sampleIndex;
}

void gs::NeuralNetwork::train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples)
{
#if NN_DEBUG
	__elapsedTime = time(0);
#endif

	this->__numTrainingSample = numTrainingSamples;
	createBuffers();
	createTrainingBuffers(trainingSetInput, trainingSetOutput, numTrainingSamples);
	addNNKernelArg();
	readBuffers();

	srand(time(NULL));

	int* trainingSamples = new int[STOCHASTIC_SAMPLING_SIZE];
	const int printIteration = 100;
	printf("\n\nTraining...\n" );

	for (int stochSampIndex = 0; stochSampIndex < SAMPLING_ITERATIONS; stochSampIndex++)
	{
		for (int tsIndex = 0; tsIndex < STOCHASTIC_SAMPLING_SIZE; tsIndex++)
		{
			float r = ((float)__numTrainingSample - 1.0f)*((float)rand() / (float)RAND_MAX);
			trainingSamples[tsIndex] = (int)floor(r);
		}

		/*Clear Buffers*/
		clearGradient();
		clearCost();

		/*Set the sample indices*/
		setImageIndex(trainingSamples);

		/*Calculate the Activations*/
		calculateActivationsLayer1();
		calculateActivationsLayer2();
		calculateActivationsLayerOutput();

		/*Calculate the Activation Deltas*/
		calculateActivationsDeltaLayerOutput();
		calculateActivationsDeltaLayer2();
		calculateActivationsDeltaLayer1();

		/*Calculate the Gradients*/
		addGradientBiasLayerOutput();
		addGradientWeightLayerOutput();
		addGradientBiasLayer2();
		addGradientWeightLayer2();
		addGradientBiasLayer1();
		addGradientWeightLayer1();

		if (stochSampIndex % printIteration == 0)
			printf("iteration %i, ", stochSampIndex);
		
		/*Compute Cost*/
#if NN_DEBUG
		computeCost();
		normalizeGradient();
		readCost();
		__cost = __cost / (float)STOCHASTIC_SAMPLING_SIZE;
		__miniBatchCostHistory.push_back(__cost);

		if (stochSampIndex % printIteration == 0)
			printf("mini-batch cost: %0.6f", __cost);

#endif

		if (stochSampIndex % printIteration == 0)
			printf("\n");

		updateNNParams();

	}
	delete trainingSamples;
#if NN_DEBUG
	__elapsedTime = time(0) - __elapsedTime;
#endif
}

float gs::NeuralNetwork::totalCost()
{
	clearCost();
	int imageIndex = 0;
	int* trainingSetIndex = new int[STOCHASTIC_SAMPLING_SIZE];

	int numIters = __numTrainingSample / STOCHASTIC_SAMPLING_SIZE;

	for (int iterIndex = 0; iterIndex < numIters; iterIndex++)
	{
		for (int i = 0; i < STOCHASTIC_SAMPLING_SIZE; i++)
		{
			trainingSetIndex[i] = imageIndex;
			imageIndex++;
		}

		setImageIndex(trainingSetIndex);
		calculateActivationsLayer1();
		calculateActivationsLayer2();
		calculateActivationsLayerOutput();
		computeCost();
	}
	float totalCost;
	readCost();
	totalCost = __cost;

	int remainder = __numTrainingSample % STOCHASTIC_SAMPLING_SIZE;
	for (int i = 0; i < remainder; i++)
	{
		trainingSetIndex[i] = imageIndex;
		imageIndex++;
	}

	setImageIndex(trainingSetIndex);
	calculateActivationsLayer1();
	calculateActivationsLayer2();
	calculateActivationsLayerOutput();
	readActivations();

	for (int i = 0; i < remainder; i++)
	{
		for (int j = 0; j < NN_OUTPUT_SIZE; j++)
		{
			float outTruth = __trainingLabelsVector[trainingSetIndex[i] * NN_OUTPUT_SIZE + j];
			float outAct = __activations[i*NN_ACTIVATION_SIZE + NN_LAYER_1_SIZE + NN_LAYER_2_SIZE + j];
			float err = pow((outTruth - outAct), 2.0);
			totalCost += err;
		}
	}
	delete trainingSetIndex;
	return totalCost / (float)__numTrainingSample;
}

void gs::NeuralNetwork::predict(float* inputVector, float* outputVector, size_t numSamples)
{
	float* in;
	size_t ns;

	if (__memobjInputVector == nullptr || __memobjOutputTruthVector == nullptr)
		createTrainingBuffers();

	for (size_t iter = 0; iter < numSamples; iter = iter + STOCHASTIC_SAMPLING_SIZE)
	{
		if (iter + STOCHASTIC_SAMPLING_SIZE > numSamples)
			ns = numSamples % STOCHASTIC_SAMPLING_SIZE;
		else
			ns = STOCHASTIC_SAMPLING_SIZE;

		in = &(inputVector[iter*NN_INPUT_SIZE]);

		cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjInputVector, CL_TRUE, 0, ns*NN_INPUT_SIZE*sizeof(float), in, 0, NULL, NULL);
		for (int i = 0; i < ns; i++)
		{
			__sampleIndex[i] = i;
		}
		setImageIndex(__sampleIndex);
		calculateActivationsLayer1();
		calculateActivationsLayer2();
		calculateActivationsLayerOutput();
		readActivations();

		for (int actIndex = 0; actIndex < ns; actIndex++)
		{
			for (int i = 0; i < NN_OUTPUT_SIZE; i++)
			{
				size_t offset = NN_OUTPUT_SIZE*(iter + actIndex);
				size_t actOffset = NN_ACTIVATION_SIZE*actIndex + NN_LAYER_1_SIZE + NN_LAYER_2_SIZE;
				outputVector[offset + i] = __activations[actOffset + i];
			}
		}

	}
}

float* gs::NeuralNetwork::createImageVector(float* inputImage)
{
	size_t vectorSize = __numTrainingSample*NN_INPUT_SIZE;
	float* imVec = new float[vectorSize];
	for (size_t i = 0; i < vectorSize; i++)
	{
		imVec[i] = inputImage[i];
	}
	return imVec;
}

float* gs::NeuralNetwork::createOutputVector(float* trainingLabels)
{
	size_t vectorSize = __numTrainingSample*NN_OUTPUT_SIZE;
	float* outVec = new float[vectorSize];
	int outVecIndex = 0;
	for (size_t i = 0; i < vectorSize; i++)
	{
		outVec[i] = trainingLabels[i];
	}
	return outVec;
}

void gs::NeuralNetwork::createBuffers()
{
	cl_int ret;

	__nnParams = new float[NN_WEIGHT_SIZE];
	__activations = new float[STOCHASTIC_SAMPLING_SIZE*NN_ACTIVATION_SIZE];
	__activationDeltas = new float[STOCHASTIC_SAMPLING_SIZE*NN_ACTIVATION_SIZE];
	__gradient = new float[NN_WEIGHT_SIZE];

	size_t activationSize = STOCHASTIC_SAMPLING_SIZE*NN_ACTIVATION_SIZE*sizeof(float);
	size_t nnParamSize = NN_WEIGHT_SIZE*sizeof(float);

	initNNParams();

	__memobjActivationVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, NULL, &ret);
	__memobjNNParamsVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, nnParamSize, NULL, &ret);
	__memobjActivationDeltaVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, NULL, &ret);
	__memobjGradientVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, nnParamSize, NULL, &ret);
	__memobjCost = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
	__memobjTrainingIndex = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, sizeof(int)*STOCHASTIC_SAMPLING_SIZE, NULL, &ret);

	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjNNParamsVector, CL_TRUE, 0, nnParamSize, __nnParams, 0, NULL, NULL);
	__cost = 0.0;
	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjCost, CL_TRUE, 0, sizeof(float), &__cost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	initGradientVector();
}

void gs::NeuralNetwork::createTrainingBuffers(float* inputImage, float* outputVector, size_t numTrainingSamples)
{
	int ret;

	size_t inputBufferSize = numTrainingSamples*NN_INPUT_SIZE*sizeof(float);
	size_t outputVectorSize = numTrainingSamples*NN_OUTPUT_SIZE*sizeof(float);

	float* inputImageVec = createImageVector(inputImage);
	float* outputImageVec = createOutputVector(outputVector);

	__memobjInputVector = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &ret);
	__memobjOutputTruthVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, outputVectorSize, NULL, &ret);

	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjInputVector, CL_TRUE, 0, inputBufferSize, inputImageVec, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjOutputTruthVector, CL_TRUE, 0, outputVectorSize, outputImageVec, 0, NULL, NULL);

	__inputImageVector = inputImageVec;
	__trainingLabelsVector = outputImageVec;
}

void gs::NeuralNetwork::createTrainingBuffers()
{
	int ret;

	size_t inputBufferSize = STOCHASTIC_SAMPLING_SIZE*sizeof(float);
	size_t outputBufferSize = STOCHASTIC_SAMPLING_SIZE*sizeof(float);

	__memobjInputVector = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &ret);
	__memobjOutputTruthVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, outputBufferSize, NULL, &ret);

	__inputImageVector = nullptr;
	__trainingLabelsVector = nullptr;
}

void gs::NeuralNetwork::initNNParams()
{
	srand(time(NULL));

	float randRange = sqrt(6.0f / (float)(NN_INPUT_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_1_WEIGHT_SIZE; i++)
	{
		__nnParams[i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	randRange = sqrt(6.0f / (float)(NN_LAYER_2_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_2_WEIGHT_SIZE; i++)
	{
		__nnParams[NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	randRange = sqrt(6.0f / (float)(NN_OUTPUT_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_OUTPUT_WEIGHT_SIZE; i++)
	{
		__nnParams[NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE + i]
			= randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	for (int i = 0; i < NN_LAYER_1_BIAS_SIZE; i++)
	{
		__nnParams[NN_LAYER_1_WEIGHT_SIZE + i] = 0.0;
	}

	for (int i = 0; i < NN_LAYER_2_BIAS_SIZE; i++)
	{
		__nnParams[NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + i] = 0.0;
	}

	for (int i = 0; i < NN_LAYER_OUTPUT_BIAS_SIZE; i++)
	{
		__nnParams[NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE + NN_LAYER_OUTPUT_WEIGHT_SIZE + i] = 0.0;
	}
}

void gs::NeuralNetwork::initGradientVector()
{
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		__gradient[i] = 0.0;
	}
	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), __gradient, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void gs::NeuralNetwork::addNNKernelArg()
{
	cl_int numSampleTemp = 0;

	/* ACTIVATION LAYER 1*/
	addKernelArg(0, 0, sizeof(cl_mem), (void*)&__memobjInputVector);
	addKernelArg(0, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(0, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(0, 3, sizeof(cl_mem), (void*)&__memobjNNParamsVector);

	/* ACTIVATION LAYER 2*/
	addKernelArg(1, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(1, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);

	/* ACTIVATION LAYER OUTPUT */
	addKernelArg(2, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(2, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);



	/* ACTIVATION DELTA LAYER 1*/
	addKernelArg(3, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(3, 1, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(3, 2, sizeof(cl_mem), (void*)&__memobjNNParamsVector);


	/* ACTIVATION DELTA LAYER 2*/
	addKernelArg(4, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(4, 1, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(4, 2, sizeof(cl_mem), (void*)&__memobjNNParamsVector);

	/* ACTIVATION DELTA LAYER OUTPUT*/
	addKernelArg(5, 0, sizeof(cl_mem), (void*)&__memobjOutputTruthVector);
	addKernelArg(5, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(5, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(5, 3, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);



	/* GRADIENT WEIGHT LAYER 1*/
	addKernelArg(6, 0, sizeof(cl_mem), (void*)&__memobjInputVector);
	addKernelArg(6, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(6, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(6, 3, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(6, 4, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(6, 5, sizeof(cl_mem), (void*)&__memobjGradientVector);


	/* GRADIENT WEIGHT LAYER 2*/
	addKernelArg(7, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(7, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(7, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(7, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);


	/* GRADIENT WEIGHT LAYER OUTPUT*/
	addKernelArg(8, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(8, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(8, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(8, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);


	/* GRADIENT BIAS LAYER 1*/
	addKernelArg(9, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(9, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(9, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(9, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/* GRADIENT BIAS LAYER 2*/
	addKernelArg(10, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(10, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(10, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(10, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/* GRADIENT BIAS LAYER OUTPUT*/
	addKernelArg(11, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(11, 1, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(11, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(11, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*UPDATE VECTOR*/
	addKernelArg(12, 0, sizeof(cl_mem), (void*)&__memobjNNParamsVector);
	addKernelArg(12, 1, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*COST FUNCTION*/
	addKernelArg(13, 0, sizeof(cl_mem), (void*)&__memobjOutputTruthVector);
	addKernelArg(13, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(13, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(13, 3, sizeof(cl_mem), (void*)&__memobjCost);

	/*NORMALIZE GRADIENT*/
	addKernelArg(14, 0, sizeof(cl_mem), (void*)&__memobjGradientVector);
	int trainingSample = STOCHASTIC_SAMPLING_SIZE;
	addKernelArg(14, 1, sizeof(cl_int), (void*)&trainingSample);

}

void gs::NeuralNetwork::exportNNParams(char* filePath)
{
	//Neural network sizes are assumed to be the size they are defined.
	readBuffers();
	std::fstream file;
	file.open(filePath, std::fstream::out | std::fstream::binary);
	if (!file)
		return;

	file.write((char*)&__inputSize, sizeof(size_t));
	file.write((char*)&__activationLayer1Size, sizeof(size_t));
	file.write((char*)&__activationLayer2Size, sizeof(size_t));
	file.write((char*)&__activationOutputSize, sizeof(size_t));
	file.write((char*)&__learningRate, sizeof(float));
	file.write((char*)&__minibatchSize, sizeof(size_t));
	file.write((char*)&__epochs, sizeof(size_t));

	//write the weights to a file.
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		float w = __nnParams[i];
		file.write((char*)&w, 4);
	}

	file.close();
}

void gs::NeuralNetwork::importNNParams(char* filePath)
{
	std::fstream file;
	file.open(filePath, std::fstream::in | std::fstream::binary);

	if (!file)
		return;

	file.read((char*)&__inputSize, sizeof(size_t));
	file.read((char*)&__activationLayer1Size, sizeof(size_t));
	file.read((char*)&__activationLayer2Size, sizeof(size_t));
	file.read((char*)&__activationOutputSize, sizeof(size_t));
	file.read((char*)&__learningRate, sizeof(float));
	file.read((char*)&__minibatchSize, sizeof(size_t));
	file.read((char*)&__epochs, sizeof(size_t));

	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		float w;
		file.read((char*)&w, 4);
		__nnParams[i] = w;
	}
	file.close();
	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjNNParamsVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), __nnParams, 0, NULL, NULL);
}

#if NN_DEBUG

void gs::NeuralNetwork::exportReport(char* filePath)
{
	time_t t = time(0);

	std::string fp(filePath);
	std::ofstream myfile;
	myfile.open(fp, std::ofstream::out);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";


	myfile << "Training time: ";
	myfile << std::to_string((float)__elapsedTime / 60.0);
	myfile << "  min";
	myfile << "\n";


	myfile << "input vector size: ";
	myfile << std::to_string(NN_INPUT_SIZE);
	myfile << "\n";

	myfile << "layer 1 activation size: ";
	myfile << std::to_string(NN_LAYER_1_SIZE);
	myfile << "\n";

	myfile << "layer 2 activation size: ";
	myfile << std::to_string(NN_LAYER_2_SIZE);
	myfile << "\n";

	myfile << "output vector size: ";
	myfile << std::to_string(NN_OUTPUT_SIZE);
	myfile << "\n";

	myfile << "stochastic minibatch size: ";
	myfile << std::to_string(STOCHASTIC_SAMPLING_SIZE);
	myfile << "\n";

	myfile << "minibatch cost history:\n";
	for (int i = 0; i < __miniBatchCostHistory.size(); i++)
	{
		myfile << std::to_string(__miniBatchCostHistory[i]);
		myfile << "\n";
	}

	float c = totalCost();
	myfile << "total training set cost :";
	myfile << std::to_string(c);

	myfile.close();
}

#endif

std::vector<std::string> gs::NeuralNetwork::kernelGlobals(int inputSize, int layer1Size, int layer2Size, int outputSize, float learningRate, int minibatchSize)
{
	std::vector<std::string> defines;

	std::string defInput("#define INPUT_SIZE ");
	defInput.append(std::to_string(inputSize));
	defInput.append(" \n");

	std::string defLayer1Size("#define LAYER_1_SIZE ");
	defLayer1Size.append(std::to_string(layer1Size));
	defLayer1Size.append(" \n");

	std::string defLayer2Size("#define LAYER_2_SIZE ");
	defLayer2Size.append(std::to_string(layer2Size));
	defLayer2Size.append(" \n");

	std::string defLayerOutputSize("#define OUTPUT_SIZE ");
	defLayerOutputSize.append(std::to_string(outputSize));
	defLayerOutputSize.append(" \n");

	std::string defLearningRate("#define LEARNING_RATE ");
	defLearningRate.append(std::to_string(learningRate));
	defLearningRate.append(" \n");

	std::string defMinibatchSize("#define STOCHASTIC_SAMPLING_SIZE ");
	defMinibatchSize.append(std::to_string(minibatchSize));
	defMinibatchSize.append(" \n");


	defines.push_back(defInput);
	defines.push_back(defLayer1Size);
	defines.push_back(defLayer2Size);
	defines.push_back(defLayerOutputSize);
	defines.push_back(defLearningRate);
	defines.push_back(defMinibatchSize);

	return defines;
}


gs::ClKernel::ClKernel(const char* kernelSource, ClContext* context, std::vector<std::string> &kernelDefines)
{
	this->__context = context;
	FILE *fp;
	char *source_str;
	size_t source_size;

	fopen_s(&fp, kernelSource, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	fseek(fp, 0, SEEK_END);
	long fSize = ftell(fp);
	rewind(fp);

	long kernelDefinesSize = 0;
	for (int i = 0; i < kernelDefines.size(); i++)
	{
		kernelDefinesSize += kernelDefines[i].size();
	}
	size_t bufferSize = kernelDefinesSize + fSize;

	source_str = (char*)malloc(bufferSize);
	source_size = fread(source_str + kernelDefinesSize, 1, fSize, fp);
	fclose(fp);
	cl_int ret;

	int stringIndex = 0;
	for (int i = 0; i < kernelDefines.size(); i++)
	{
		for (int j = 0; j < kernelDefines[i].size(); j++)
		{
			source_str[stringIndex] = (kernelDefines[i])[j];
			stringIndex++;
		}
	}
	bufferSize = source_size + kernelDefinesSize;
	__program = NULL;
	__program = clCreateProgramWithSource(context->context, 1, (const char**)&source_str, (const size_t*)&bufferSize, &ret);
	switch (ret)
	{
	case CL_INVALID_CONTEXT:
	{
		printf("ERROR, context is invalid.\n");
		break;
	}
	case CL_INVALID_VALUE:
	{
		printf("ERROR, invalid value.\n");
		break;
	}
	case CL_OUT_OF_HOST_MEMORY:
	{
		printf("ERROR, out of host memory. \n");
		break;
	}
	case CL_SUCCESS:
	{
		printf("Success, program created. \n");
	}
	}

	ret = clBuildProgram(__program, 1, &context->deviceId, NULL, NULL, NULL);
	switch (ret)
	{
	case CL_INVALID_PROGRAM:
	{
		printf("ERROR, program is invalid. \n");
		break;
	}
	case CL_INVALID_VALUE:
	{
		printf("ERROR, invalid value.\n");
		break;
	}
	case CL_INVALID_DEVICE:
	{
		printf("ERROR, invalid device.\n");
		break;
	}
	case CL_INVALID_BINARY:
	{
		printf("ERROR, invalid binary program.\n");
		break;
	}
	case CL_INVALID_BUILD_OPTIONS:
	{
		printf("ERROR, invalid build options.\n");
		break;
	}
	case CL_INVALID_OPERATION:
	{
		printf("ERROR, invalid build operation.\n");
		break;
	}
	case CL_COMPILER_NOT_AVAILABLE:
	{
		printf("ERROR, compiler not available.\n");
		break;
	}
	case CL_OUT_OF_HOST_MEMORY:
	{
		printf("ERROR, out of host memory.\n");
		break;
	}
	case CL_BUILD_PROGRAM_FAILURE:
	{
		printf("ERROR, build failed.\n");
		break;
	}
	case CL_SUCCESS:
	{
		printf("Success, build successful.\n");
		break;
	}

	}

	size_t buildLogSize;
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
	char* buildLog = new char[buildLogSize];
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);

	printf(buildLog);
	printf("\n");
	delete buildLog;

	free(source_str);

	__globalWorkSize = 0;
	__localWorkSize = 0;
}
gs::ClKernel::~ClKernel()
{
	for (int i = 0; i < __kernel.size(); i++)
	{
		clReleaseKernel(__kernel[i]);
	}
	__kernel.clear();

	clReleaseProgram(__program);
	for (int i = 0; i < __memoryObjects.size(); i++)
	{
		clReleaseMemObject(__memoryObjects[i]);
	}
	__memoryObjects.clear();
}

void gs::ClKernel::train()
{

}

int gs::ClKernel::createKernel(const char* kernelName)
{
	cl_int ret;
	cl_kernel k = clCreateKernel(__program, kernelName, &ret);
	int kernelPos = -1;

	switch (ret)
	{
	case CL_INVALID_PROGRAM:
		printf("ERROR, program is not a valid program object. \n");
		break;

	case CL_INVALID_PROGRAM_EXECUTABLE:
		printf("ERROR, there is no successfully built executable for program. \n");
		break;

	case CL_INVALID_KERNEL_NAME:
		printf("ERROR, 'kernelName' is not found in the program. \n");
		break;

	case CL_INVALID_KERNEL_DEFINITION:
		printf("ERROR, invalid kernel definition. \n");
		break;

	case CL_INVALID_VALUE:
		printf("ERROR, 'kernelName' is NULL. \n");
		break;

	case CL_OUT_OF_HOST_MEMORY:
		printf("ERROR, out of host memory. \n");
		break;

	case CL_SUCCESS:
		printf("Success, kernel created. \n");
		__kernel.push_back(k);
		kernelPos = __kernel.size();
		break;
	}

	char kernelInfo[1000];
	size_t kernelInfoSize;

	clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 1000, kernelInfo, &kernelInfoSize);
	printf("Kernel function name: ");
	printf(kernelInfo);
	printf("\n");

	return kernelPos;
}

void gs::ClKernel::addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer)
{
	if (__kernel.size() <= kernelIndex)
	{
		printf("ERROR, kernel out of index. \n");
		return;
	}
	cl_int ret = clSetKernelArg(__kernel[kernelIndex], argId, bufferSize, buffer);

	switch (ret)
	{
	case CL_INVALID_KERNEL:
		printf("ERROR, invalid kernel object. \n");
		break;

	case CL_INVALID_ARG_INDEX:
		printf("ERROR, 'argId' is not a valid argument index. \n");
		break;

	case CL_INVALID_ARG_VALUE:
		printf("ERROR, 'bufer' is NULL. \n");
		break;

	case CL_INVALID_MEM_OBJECT:
		printf("ERROR, memory buffer is not a valid memory object. \n");
		break;

	case CL_INVALID_SAMPLER:
		printf("ERROR, sampler buffer is not a valid sampler object. \n");
		break;

	case CL_INVALID_ARG_SIZE:
		printf("ERROR, invalid argument size. \n");
		break;

	case CL_SUCCESS:
		//printf("Success, kernel argument added. \n");
		break;

	}
}

void gs::ClKernel::createBuffer(cl_context context, cl_mem_flags memFlag, size_t bufferSize, cl_mem &memObj)
{
	cl_int ret;
	memObj = clCreateBuffer(context, memFlag, bufferSize, NULL, &ret);

	switch (ret)
	{
	case CL_SUCCESS:

		break;

	case CL_INVALID_CONTEXT:
		printf("ERROR, context is not valid");
		break;

	case CL_INVALID_VALUE:
		printf("ERROR, memory flags are not valid");
		break;

	case CL_INVALID_BUFFER_SIZE:
		printf("ERROR, invalid buffer size");
		break;

	case CL_INVALID_HOST_PTR:
		printf("ERROR, host pointer is invalid");
		break;

	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		printf("ERROR, memory allocation failure");
		break;

	case CL_OUT_OF_HOST_MEMORY:
		printf("ERROR, out of host memory");
		break;
	}
}


gs::ClContext::ClContext()
{
	deviceId = NULL;
	platformId = NULL;
	context = NULL;
	commandQueue = NULL;

	createContext();
	createQueue();
}
gs::ClContext::~ClContext()
{
	cl_int ret;
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
}

void gs::ClContext::createContext()
{
	cl_int ret;
	ret = clGetPlatformIDs(2, &platformId, &retNumPlatforms);

	switch (ret)
	{
	case CL_SUCCESS:
	{
		printf("%i platform(s) found: \n", retNumPlatforms);
		printPlatformInfo(platformId, retNumPlatforms);
		break;
	}
	case CL_INVALID_VALUE:
	{
		printf("ERROR, could not find valid platforms.\n");
		return;
	}
	default:
		break;
	}

	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, &retNumDevices);
	switch (ret)
	{
	case CL_SUCCESS:
	{
		printf("Success, device found. \n");
		printDeviceInfo(deviceId);
		break;
	}

	case CL_INVALID_PLATFORM:
	{
		printf("ERROR, platform is not a valid platform.\n");
		break;
	}

	case CL_INVALID_DEVICE_TYPE:
	{
		printf("ERROR, num_entries is equal to zero and device_type is not NULL or both num_devices and device_type are NULL \n");
		break;
	}

	case CL_INVALID_VALUE:
	{
		printf("ERROR, no OpenCL devices that matched device_type were found \n");
		break;
	}

	case CL_DEVICE_NOT_FOUND:
	{
		printf("ERROR, device not found \n");
		break;
	}
	}

	context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);
}
void gs::ClContext::createQueue()
{
	cl_int ret;
	commandQueue = clCreateCommandQueue(context, deviceId, 0, &ret);
}

void gs::ClContext::printPlatformInfo(cl_platform_id platformId, cl_uint retNumPlatforms)
{
	cl_int retPlat;
	size_t returnStringSize;
	char stringRet[128];

	for (int i = 0; i < retNumPlatforms; i++)
	{
		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_PROFILE, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");
	}
}

void gs::ClContext::printDeviceInfo(cl_device_id deviceId)
{
	size_t returnedSize;

	cl_int addressBits;
	cl_ulong cacheSize;
	cl_device_mem_cache_type cacheType;
	cl_uint cacheLineSize;
	cl_ulong globalMemSize;

	char deviceName[128];
	char deviceVendor[128];
	char deviceVersion[128];

	clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 128, &deviceName, &returnedSize);
	printf(deviceName);
	printf("\n");

	clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, 128, &deviceVendor, &returnedSize);
	printf(deviceVendor);
	printf("\n");

	clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 128, &deviceVersion, &returnedSize);
	printf(deviceVersion);
	printf("\n");


	clGetDeviceInfo(deviceId, CL_DEVICE_ADDRESS_BITS, sizeof(cl_int), &addressBits, &returnedSize);
	printf("address bits: %i \n", addressBits);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheSize, &returnedSize);
	printf("Size of global memory cache in bytes: %i \n", cacheSize);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &cacheLineSize, &returnedSize);
	printf("Size of global memory cache line in bytes: %u \n", cacheLineSize);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, &returnedSize);
	printf("Size of global device memory in bytes: %X \n", globalMemSize);


	char extensions[128];
	clGetDeviceInfo(deviceId, CL_DEVICE_EXTENSIONS, 128, &extensions, &returnedSize);
	printf("extensions: \n");
	printf(extensions);

	cl_bool imageSupport;
	clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, &returnedSize);
	if (imageSupport == CL_FALSE)
		printf("images are not supported \n");
	else if (imageSupport == CL_TRUE)
	{
		printf("images are supported \n");

		size_t image2DMaxHeight, image2DMaxwidth;
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &image2DMaxwidth, &returnedSize);
		printf("Max width of 2D image in pixels: %u \n", image2DMaxwidth);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &image2DMaxHeight, &returnedSize);
		printf("Max height of 2D image in pixels: %u \n", image2DMaxHeight);

		size_t image3DMaxHeight, image3DMaxwidth, image3DMaxDepth;
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &image3DMaxwidth, &returnedSize);
		printf("Max width of 3D image in pixels: %u \n", image3DMaxwidth);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &image3DMaxHeight, &returnedSize);
		printf("Max height of 3D image in pixels: %u \n", image3DMaxHeight);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &image3DMaxDepth, &returnedSize);
		printf("Max depth of 3D image in pixels: %u \n", image3DMaxDepth);

		cl_uint maxSamplers;
		clGetDeviceInfo(deviceId, CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint), &maxSamplers, &returnedSize);
		printf("Maximum number of samplers that can be used in a kernel. %u \n", maxSamplers);

	}
	cl_ulong localMemSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, &returnedSize);
	printf("Size of local memory arena in bytes: %u \n", localMemSize);

	cl_uint maxClockFreq;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFreq, &returnedSize);
	printf("Maximum configured clock frequency of the device: %u \n", maxClockFreq);

	cl_uint maxComputeUnits;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &returnedSize);
	printf("The number of parallel compute cores on the OpenCL device %u \n", maxComputeUnits);

	cl_uint maxConstantArgs;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &maxConstantArgs, &returnedSize);
	printf("Max number of arguments declared with the __constant qualifier in a kernel %u \n", maxConstantArgs);

	cl_ulong maxConstantBufferSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &maxConstantBufferSize, &returnedSize);
	printf("Max size in bytes of a constant buffer allocation. %X \n", maxConstantBufferSize);

	cl_ulong maxMemAllocationSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocationSize, &returnedSize);
	printf("Max size of memory object allocation in bytes. %X \n", maxMemAllocationSize);

	size_t maxParamSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &maxParamSize, &returnedSize);
	printf("Max size in bytes of the arguments that can be passed to a kernel. %i \n", maxParamSize);


	size_t maxWorkgroupSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkgroupSize, &returnedSize);
	printf("Maximum number of work-items in a work-group executing a kernel using the data parallel execution model. %u \n", maxWorkgroupSize);

	size_t maxWorkItemSize[3];
	maxWorkItemSize[0] = 0;
	maxWorkItemSize[1] = 0;
	maxWorkItemSize[2] = 0;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * 4, &maxWorkItemSize, &returnedSize);
	printf("Maximum number of work-items that can be specified in each dimension of the work-group. %u, %u, %u \n", maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2]);

	cl_uint maxWorkItemDimensions;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &maxWorkItemDimensions, &returnedSize);
	printf("Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. %u \n", maxWorkItemDimensions);
}


gs::ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(ClContext* context, size_t inputImageWidth, size_t inputImageHeight, size_t outputSize, float learningRate, size_t minibatchSize, size_t numEpochs) : ClKernel(CNN_KENRNEL_SOURCE, context, kernelGlobals(inputImageWidth, inputImageHeight, outputSize, learningRate, minibatchSize))
{
	cl_int ret;
	createKernel("activationC1");
	createKernel("activationS2");
	createKernel("activationC3");
	createKernel("activationS4");
	createKernel("activationC5");
	createKernel("activationF6");
	createKernel("activationOutput");


	createKernel("activationDeltaC1");
	createKernel("activationDeltaS2");
	createKernel("activationDeltaC3");
	createKernel("activationDeltaS4");
	createKernel("activationDeltaC5");
	createKernel("activationDeltaF6");
	createKernel("activationDeltaOutput");


	createKernel("addGradientWeightC1");
	createKernel("addGradientWeightS2");
	createKernel("addGradientWeightC3");
	createKernel("addGradientWeightS4");
	createKernel("addGradientWeightC5");
	createKernel("addGradientWeightF6");
	createKernel("addGradientWeightOutput");


	createKernel("addGradientBiasC1");
	createKernel("addGradientBiasS2");
	createKernel("addGradientBiasC3");
	createKernel("addGradientBiasS4");
	createKernel("addGradientBiasC5");
	createKernel("addGradientBiasF6");
	createKernel("addGradientBiasOutput");


	createKernel("normalizeGradient");
	createKernel("updateNNparams");
	createKernel("cost");

	totalWorkItems = 0;

	this->__inputWidth = inputImageWidth;
	this->__inputHeight = inputImageHeight;
	this->__learningRate = learningRate;
	this->__epochs = numEpochs;
	this->__minibatchSize = minibatchSize;
	this->__activationOutputSize = outputSize;

	__cnnParams = nullptr;
	__activations = nullptr;
	__activationDeltas = nullptr;
	__gradient = nullptr;

	__trainingInputVector = nullptr;
	__trainingOutputVector = nullptr;
	__sampleIndex = new int[CNN_STOCHASTIC_SAMPLING_SIZE];

	createBuffers();

	/*need to add appropriate args*/
	addCNNKernelArg();
	readBuffers();
}


gs::ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork()
{
	clReleaseMemObject(__memobjInputVector);
	clReleaseMemObject(__memobjOutputTruthVector);
	clReleaseMemObject(__memobjActivationVector);
	clReleaseMemObject(__memobjCNNParamsVector);
	clReleaseMemObject(__memobjActivationDeltaVector);
	clReleaseMemObject(__memobjGradientVector);
	clReleaseMemObject(__memobjCost);

	delete __cnnParams;
	delete __activations;
	delete __activationDeltas;
	delete __gradient;

	delete __trainingInputVector;
	delete __trainingOutputVector;
}

void gs::ConvolutionalNeuralNetwork::train(float* trainingSetInput, float* trainingSetOutput, size_t numTrainingSamples)
{
#if NN_DEBUG
	__elapsedTime = time(0);
#endif

	srand(time(NULL));
	createTrainingBuffers(trainingSetInput, trainingSetOutput, numTrainingSamples);
	addCNNKernelArg();
	readBuffers();
	const int printIteration = 200;
	int* trainingSamples = new int[CNN_STOCHASTIC_SAMPLING_SIZE];

	for (int stochSampIndex = 0; stochSampIndex < CNN_SAMPLING_ITERATIONS; stochSampIndex++)
	{
		for (int tsIndex = 0; tsIndex < CNN_STOCHASTIC_SAMPLING_SIZE; tsIndex++)
		{
			float r = ((float)__numTrainingSample - 1.0f)*((float)rand() / (float)RAND_MAX);
			trainingSamples[tsIndex] = (int)floor(r);
		}

		clearGradient();
		clearCost();

		setImageIndex(trainingSamples);
		calculateActivationsC1();
		calculateActivationsS2();
		calculateActivationsC3();
		calculateActivationsS4();
		calculateActivationsC5();
		calculateActivationsF6();
		calculateActivationsOutput();

		calculateActivationDeltaOutput();
		calculateActivationDeltaF6();
		calculateActivationDeltaC5();
		calculateActivationDeltaS4();
		calculateActivationDeltaC3();
		calculateActivationDeltaS2();
		calculateActivationDeltaC1();

		addGradientWeightC1();
		addGradientWeightS2();
		addGradientWeightC3();
		addGradientWeightS4();
		addGradientWeightC5();
		addGradientWeightF6();
		addGradientWeightOutput();

		addGradientBiasC1();
		addGradientBiasS2();
		addGradientBiasC3();
		addGradientBiasS4();
		addGradientBiasC5();
		addGradientBiasF6();


		if (stochSampIndex % printIteration == 0)
			printf("iteration %i, ", stochSampIndex);

#if NN_DEBUG

		computeCost();
		readCost();
		__cost = __cost / (float)CNN_STOCHASTIC_SAMPLING_SIZE;
		__miniBatchCostHistory.push_back(__cost);

		if (stochSampIndex % printIteration == 0)
			printf("mini-batch cost: %0.6f", __cost);
#endif

		if (stochSampIndex % printIteration == 0)
			printf("\n");
		updateNNParams();
	}


	delete trainingSamples;
#if NN_DEBUG
	__elapsedTime = time(0) - __elapsedTime;
#endif
}

float gs::ConvolutionalNeuralNetwork::totalCost()
{
	clearCost();
	int imageIndex = 0;
	int* trainingSetIndex = new int[CNN_STOCHASTIC_SAMPLING_SIZE];

	int numIters = __numTrainingSample / CNN_STOCHASTIC_SAMPLING_SIZE;

	for (int iterIndex = 0; iterIndex < numIters; iterIndex++)
	{
		for (int i = 0; i < CNN_STOCHASTIC_SAMPLING_SIZE; i++)
		{
			trainingSetIndex[i] = imageIndex;
			imageIndex++;
		}

		setImageIndex(trainingSetIndex);
		calculateActivationsC1();
		calculateActivationsS2();
		calculateActivationsC3();
		calculateActivationsS4();
		calculateActivationsC5();
		calculateActivationsF6();
		calculateActivationsOutput();
		computeCost();
	}
	float totalCost;
	readCost();
	totalCost = __cost;

	int remainder = __numTrainingSample % CNN_STOCHASTIC_SAMPLING_SIZE;
	for (int i = 0; i < remainder; i++)
	{
		trainingSetIndex[i] = imageIndex;
		imageIndex++;
	}

	setImageIndex(trainingSetIndex);
	calculateActivationsC1();
	calculateActivationsS2();
	calculateActivationsC3();
	calculateActivationsS4();
	calculateActivationsC5();
	calculateActivationsF6();
	calculateActivationsOutput();
	readActivations();

	for (int i = 0; i < remainder; i++)
	{
		for (int j = 0; j < CNN_ACTIVATION_SIZE_OUTPUT; j++)
		{
			float outTruth = __trainingOutputVector[trainingSetIndex[i] * CNN_ACTIVATION_SIZE_OUTPUT + j];
			int offset = CNN_ACTIVATION_SIZE_C_1 + CNN_ACTIVATION_SIZE_S_2 + CNN_ACTIVATION_SIZE_C_3 + CNN_ACTIVATION_SIZE_S_4 + CNN_ACTIVATION_SIZE_C_5 + CNN_ACTIVATION_SIZE_F_6;
			float outAct = __activations[i*CNN_ACTIVATION_BUFFER_SIZE + offset + j];
			float err = pow((outTruth - outAct), 2.0);
			totalCost += err;
		}
	}
	delete trainingSetIndex;
	return totalCost / (float)__numTrainingSample;
}


void gs::ConvolutionalNeuralNetwork::predict(float* inputVector, float* outputVector, size_t numSamples)
{
	float* in;
	size_t ns;

	if (__trainingInputVector == nullptr || __trainingOutputVector == nullptr)
	{
		createTrainingBuffers();
	}


	for (size_t iter = 0; iter < numSamples; iter = iter + CNN_STOCHASTIC_SAMPLING_SIZE)
	{
		if (iter + CNN_STOCHASTIC_SAMPLING_SIZE > numSamples)
			ns = numSamples % CNN_STOCHASTIC_SAMPLING_SIZE;
		else
			ns = CNN_STOCHASTIC_SAMPLING_SIZE;

		in = &(inputVector[iter*CNN_INPUT_SIZE]);

		cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjInputVector, CL_TRUE, 0, ns*CNN_INPUT_SIZE*sizeof(float), in, 0, NULL, NULL);

		for (int i = 0; i < ns; i++)
		{
			__sampleIndex[i] = i;
		}
		setImageIndex(__sampleIndex);

		calculateActivationsC1();
		calculateActivationsS2();
		calculateActivationsC3();
		calculateActivationsS4();
		calculateActivationsC5();
		calculateActivationsF6();
		calculateActivationsOutput();

		readActivations();

		for (int actIndex = 0; actIndex < ns; actIndex++)
		{
			for (int i = 0; i < CNN_ACTIVATION_SIZE_OUTPUT; i++)
			{
				size_t offset = CNN_ACTIVATION_SIZE_OUTPUT*(iter + actIndex);
				size_t actOffset = CNN_ACTIVATION_BUFFER_SIZE*actIndex + CNN_ACTIVATION_SIZE_C_1 + CNN_ACTIVATION_SIZE_S_2 + CNN_ACTIVATION_SIZE_C_3 + CNN_ACTIVATION_SIZE_S_4 + CNN_ACTIVATION_SIZE_C_5 + CNN_ACTIVATION_SIZE_F_6;
				outputVector[offset + i] = __activations[actOffset + i];
			}
		}
	}
}

float* gs::ConvolutionalNeuralNetwork::createInputVector(float* inputImage, size_t numSamples)
{
	size_t vectorSize = __numTrainingSample*__inputHeight*__inputWidth;
	float* vec = new float[vectorSize];
	for (int i = 0; i < vectorSize; i++)
	{
		vec[i] = inputImage[i];
	}
	return vec;
}
float* gs::ConvolutionalNeuralNetwork::createOutputVector(float* outputVector, size_t numSamples)
{
	size_t outVecSize = CNN_NUM_FEATURE_MAPS_OUTPUT*__numTrainingSample;
	float* outVec = new float[outVecSize];
	for (int i = 0; i < outVecSize; i++)
	{
		outVec[i] = outputVector[i];
	}
	return outVec;
}

void gs::ConvolutionalNeuralNetwork::initNNParams()
{
	srand(time(NULL));

	float randRange = 1.0;

	size_t offset = 0;
	for (int i = 0; i < CNN_C_1_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_C_1_WEIGHT_SIZE;
	for (int i = 0; i < CNN_C_1_BIAS_SIZE; i++)
	{
		__cnnParams[i] = 0.0;
	}


	offset += CNN_C_1_BIAS_SIZE;
	for (int i = 0; i < CNN_S_2_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_S_2_WEIGHT_SIZE;
	for (int i = 0; i < CNN_S_2_BIAS_SIZE; i++)
	{
		__cnnParams[offset + i] = 0.0;
	}

	offset += CNN_S_2_BIAS_SIZE;
	for (int i = 0; i < CNN_C_3_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_C_3_WEIGHT_SIZE;
	for (int i = 0; i < CNN_C_3_BIAS_SIZE; i++)
	{
		__cnnParams[offset + i] = 0.0;
	}

	offset += CNN_C_3_BIAS_SIZE;
	for (int i = 0; i < CNN_S_4_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_S_4_WEIGHT_SIZE;
	for (int i = 0; i < CNN_S_4_BIAS_SIZE; i++)
	{
		__cnnParams[offset + i] = 0.0;
	}

	offset += CNN_S_4_BIAS_SIZE;
	randRange = sqrt(6.0f / (float)(CNN_ACTIVATION_SIZE_S_4 + CNN_ACTIVATION_SIZE_C_5));
	for (int i = 0; i < CNN_C_5_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_C_5_WEIGHT_SIZE;
	for (int i = 0; i < CNN_C_5_BIAS_SIZE; i++)
	{
		__cnnParams[offset + i] = 0.0;
	}

	offset += CNN_C_5_BIAS_SIZE;
	randRange = sqrt(6.0f / (float)(CNN_ACTIVATION_SIZE_C_5 + CNN_ACTIVATION_SIZE_F_6));
	for (int i = 0; i < CNN_F_6_WEIGHT_SIZE; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	offset += CNN_F_6_WEIGHT_SIZE;
	for (int i = 0; i < CNN_F_6_BIAS_SIZE; i++)
	{
		__cnnParams[offset + i] = 0.0;
	}

	offset += CNN_F_6_BIAS_SIZE;
	randRange = sqrt(6.0f / (float)(CNN_ACTIVATION_SIZE_F_6 + CNN_ACTIVATION_SIZE_OUTPUT));
	for (int i = 0; i < CNN_OUTPUT_RBF_CENTERS; i++)
	{
		__cnnParams[offset + i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}
}

void gs::ConvolutionalNeuralNetwork::createBuffers()
{
	cl_int ret;

	__cnnParams = new float[CNN_PARAM_BUFER_SIZE];
	__activations = new float[CNN_STOCHASTIC_SAMPLING_SIZE*CNN_ACTIVATION_BUFFER_SIZE];
	__activationDeltas = new float[CNN_STOCHASTIC_SAMPLING_SIZE*CNN_ACTIVATION_BUFFER_SIZE];
	__gradient = new float[CNN_PARAM_BUFER_SIZE];


	//size_t inputBufferSize = __inputHeight*__inputHeight*__numTrainingSample*sizeof(float);
	//size_t outputVectorSize = NUM_FEATURE_MAPS_OUTPUT*__numTrainingSample*sizeof(float);
	size_t activationSize = CNN_STOCHASTIC_SAMPLING_SIZE*CNN_ACTIVATION_BUFFER_SIZE*sizeof(float);
	size_t cnnParamSize = CNN_PARAM_BUFER_SIZE*sizeof(float);

	initNNParams();

	//createBuffer(__context->context, CL_MEM_READ_WRITE, inputBufferSize, __memobjInputVector);
	//createBuffer(__context->context, CL_MEM_READ_WRITE, outputVectorSize, __memobjOutputTruthVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, __memobjActivationVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, __memobjActivationDeltaVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, cnnParamSize, __memobjCNNParamsVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, cnnParamSize, __memobjGradientVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, sizeof(float), __memobjCost);
	createBuffer(__context->context, CL_MEM_READ_WRITE, CNN_STOCHASTIC_SAMPLING_SIZE*sizeof(int), __memobjTrainingIndex);

	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjCNNParamsVector, CL_TRUE, 0, cnnParamSize, __cnnParams, 0, NULL, NULL);

	clFinish(__context->commandQueue);
}

void gs::ConvolutionalNeuralNetwork::createTrainingBuffers(float* inputVector, float* outputVector, size_t numSamples)
{
	cl_int ret;
	__numTrainingSample = numSamples;
	size_t inputBufferSize = __inputHeight*__inputHeight*__numTrainingSample*sizeof(float);
	size_t outputVectorSize = CNN_NUM_FEATURE_MAPS_OUTPUT*__numTrainingSample*sizeof(float);
	__trainingInputVector = createInputVector(inputVector, numSamples);
	__trainingOutputVector = createOutputVector(outputVector, numSamples);

	createBuffer(__context->context, CL_MEM_READ_WRITE, inputBufferSize, __memobjInputVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, outputVectorSize, __memobjOutputTruthVector);

	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjInputVector, CL_TRUE, 0, inputBufferSize, __trainingInputVector, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjOutputTruthVector, CL_TRUE, 0, outputVectorSize, __trainingOutputVector, 0, NULL, NULL);

	clFinish(__context->commandQueue);
}
void gs::ConvolutionalNeuralNetwork::createTrainingBuffers()
{
	size_t inputBufferSize = __inputHeight*__inputHeight*CNN_STOCHASTIC_SAMPLING_SIZE*sizeof(float);
	size_t outputVectorSize = CNN_NUM_FEATURE_MAPS_OUTPUT*CNN_STOCHASTIC_SAMPLING_SIZE*sizeof(float);

	createBuffer(__context->context, CL_MEM_READ_WRITE, inputBufferSize, __memobjInputVector);
	createBuffer(__context->context, CL_MEM_READ_WRITE, outputVectorSize, __memobjOutputTruthVector);

	clFinish(__context->commandQueue);
}

void gs::ConvolutionalNeuralNetwork::addCNNKernelArg()
{
	cl_int numSampleTemp = 0;

	/* ACTIVATIONS*/
	/*C1*/
	addKernelArg(0, 0, sizeof(cl_mem), (void*)&__memobjInputVector);
	addKernelArg(0, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(0, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(0, 3, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*S2*/
	addKernelArg(1, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(1, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*C3*/
	addKernelArg(2, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(2, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*S4*/
	addKernelArg(3, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(3, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*C5*/
	addKernelArg(4, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(4, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*F6*/
	addKernelArg(5, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(5, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);

	/*Output*/
	addKernelArg(6, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(6, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);


	/*ACTIVATION DELTA*/
	/*C1*/
	addKernelArg(7, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(7, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(7, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*S2*/
	addKernelArg(8, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(8, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(8, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*C3*/
	addKernelArg(9, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(9, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(9, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*S4*/
	addKernelArg(10, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(10, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(10, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*C5*/
	addKernelArg(11, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(11, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(11, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*F6*/
	addKernelArg(12, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(12, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(12, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);

	/*Output*/
	addKernelArg(13, 0, sizeof(cl_mem), (void*)&__memobjOutputTruthVector);
	addKernelArg(13, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(13, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(13, 3, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(13, 4, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);


	/*GRADIENT WEIGHT*/
	/*C1*/
	addKernelArg(14, 0, sizeof(cl_mem), (void*)&__memobjInputVector);
	addKernelArg(14, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(14, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(14, 3, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(14, 4, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(14, 5, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*S2*/
	addKernelArg(15, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(15, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(15, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(15, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*C3*/
	addKernelArg(16, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(16, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(16, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(16, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*S4*/
	addKernelArg(17, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(17, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(17, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(17, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*C5*/
	addKernelArg(18, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(18, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(18, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(18, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*F6*/
	addKernelArg(19, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(19, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(19, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(19, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*Output*/
	addKernelArg(20, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(20, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(20, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(20, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);


	/*Output Bias*/
	/*C1*/
	addKernelArg(21, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(21, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(21, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(21, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*S2*/
	addKernelArg(22, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(22, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(22, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(22, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*C3*/
	addKernelArg(23, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(23, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(23, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(23, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*S4*/
	addKernelArg(24, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(24, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(24, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(24, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*C5*/
	addKernelArg(25, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(25, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(25, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(25, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*F6*/
	addKernelArg(26, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(26, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(26, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(26, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*Output*/
	addKernelArg(27, 0, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(27, 1, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(27, 2, sizeof(cl_mem), (void*)&__memobjActivationDeltaVector);
	addKernelArg(27, 3, sizeof(cl_mem), (void*)&__memobjGradientVector);


	/*Normalize Gradient*/
	cl_int stochSamples = CNN_STOCHASTIC_SAMPLING_SIZE;
	addKernelArg(28, 0, sizeof(cl_mem), (void*)&__memobjGradientVector);
	addKernelArg(28, 1, sizeof(cl_int), (void*)&stochSamples);

	/*Update NN Params*/
	addKernelArg(29, 0, sizeof(cl_mem), (void*)&__memobjCNNParamsVector);
	addKernelArg(29, 1, sizeof(cl_mem), (void*)&__memobjGradientVector);

	/*Cost*/
	addKernelArg(30, 0, sizeof(cl_mem), (void*)&__memobjOutputTruthVector);
	addKernelArg(30, 1, sizeof(cl_mem), (void*)&__memobjTrainingIndex);
	addKernelArg(30, 2, sizeof(cl_mem), (void*)&__memobjActivationVector);
	addKernelArg(30, 3, sizeof(cl_mem), (void*)&__memobjCost);
}



void gs::ConvolutionalNeuralNetwork::exportNNParams(char* filePath)
{
	//Neural network sizes are assumed to be the size they are defined.
	readBuffers();
	std::fstream file;
	file.open(filePath, std::fstream::out | std::fstream::binary | std::fstream::trunc);
	if (!file)
		return;

	//write the weights to a file.
	for (int i = 0; i < CNN_PARAM_BUFER_SIZE; i++)
	{
		float w = __cnnParams[i];
		file.write((char*)&w, 4);
	}

	file.close();
}

void gs::ConvolutionalNeuralNetwork::importNNParams(char* filePath)
{
	std::fstream file;
	file.open(filePath, std::fstream::in | std::fstream::binary);

	if (!file)
		return;

	for (int i = 0; i < CNN_PARAM_BUFER_SIZE; i++)
	{
		float w;
		file.read((char*)&w, 4);
		__cnnParams[i] = w;
	}
	file.close();
	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, __memobjCNNParamsVector, CL_TRUE, 0, CNN_PARAM_BUFER_SIZE*sizeof(float), __cnnParams, 0, NULL, NULL);
}

void gs::ConvolutionalNeuralNetwork::exportReport(char* filePath)
{
	time_t t = time(0);

	std::string fp(filePath);
	std::ofstream myfile;
	myfile.open(fp, std::ofstream::app);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";

#if NN_DEBUG
	myfile << "Training time: ";
	myfile << std::to_string((float)__elapsedTime / 60.0);
	myfile << "  min";
	myfile << "\n";
#endif
	
	myfile << "input image width: ";
	myfile << std::to_string(CNN_IMAGE_WIDTH);
	myfile << "\n";

	myfile << "input image width: ";
	myfile << std::to_string(CNN_IMAGE_HEIGHT);
	myfile << "\n";

	myfile << "output vector size: ";
	myfile << std::to_string(CNN_NUM_FEATURE_MAPS_OUTPUT);
	myfile << "\n";

	myfile << "stochastic minibatch size: ";
	myfile << std::to_string(STOCHASTIC_SAMPLING_SIZE);
	myfile << "\n";
	
#if NN_DEBUG
	myfile << "minibatch cost history:\n";
	for (int i = 0; i < __miniBatchCostHistory.size(); i++)
	{
		myfile << std::to_string(__miniBatchCostHistory[i]);
		myfile << "\n";
	}
#endif

	myfile.close();
}

std::vector<std::string> gs::ConvolutionalNeuralNetwork::kernelGlobals(size_t inputWidth, size_t inputHeight, size_t outputSize, float learningRate, int minibatchSize)
{
	std::vector<std::string> defines;

	std::string defInputW("#define IMAGE_WIDTH ");
	defInputW.append(std::to_string(inputWidth));
	defInputW.append(" \n");

	std::string defInputH("#define IMAGE_HEIGHT ");
	defInputH.append(std::to_string(inputHeight));
	defInputH.append(" \n");

	std::string defLayerOutputSize("#define NUM_FEATURE_MAPS_OUTPUT ");
	defLayerOutputSize.append(std::to_string(outputSize));
	defLayerOutputSize.append(" \n");

	std::string defLearningRate("#define LEARNING_RATE ");
	defLearningRate.append(std::to_string(learningRate));
	defLearningRate.append(" \n");

	std::string defMinibatchSize("#define STOCHASTIC_SAMPLING_SIZE ");
	defMinibatchSize.append(std::to_string(minibatchSize));
	defMinibatchSize.append(" \n");


	defines.push_back(defInputW);
	defines.push_back(defInputH);
	defines.push_back(defLayerOutputSize);
	defines.push_back(defLearningRate);
	defines.push_back(defMinibatchSize);

	return defines;
}