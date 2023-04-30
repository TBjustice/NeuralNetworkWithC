#ifndef CNEURAL_LAYER
#define CNEURAL_LAYER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "safememory.h"

float rand_uniform(float a, float b) {
	float x = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
	return (b - a) * x + a;
}
float rand_normal(float mu, float sigma) {
	float z = sqrtf(-2.0f * logf(rand_uniform(0.0f, 1.0f))) * sinf(2.0f * 3.141592653589793f * rand_uniform(0.0f, 1.0f));
	return mu + sigma * z;
}

/* Layer */

typedef struct Layer {
	float* data;
	float* delta;
	size_t size;
} Layer_t;
void initLayer(Layer_t* layer, size_t size) {
	layer->size = size;
	layer->data = (float*)SafeMemoryAllocate(size * 2 * sizeof(float));
	layer->delta = layer->data + size;
}
void showLayerInfo(Layer_t* layer) {
	printf("data:");
	for(size_t i=0;i< layer->size; ++i) printf("%f,", layer->data[i]);
	printf("\ndelta:");
	for (size_t i = 0; i < layer->size; ++i) printf("%f,", layer->delta[i]);
	printf("\n");
}

/* Nodes */

typedef struct Nodes {
	size_t structSize;
	Layer_t* in;
	Layer_t* out;
	void (*forward)(struct Nodes*);
	void (*backward)(struct Nodes*);
	void (*fit)(struct Nodes*, float l);
} Nodes_t;
void copyForward(Nodes_t* nodes) {
	size_t size = nodes->in->size < nodes->out->size ? nodes->in->size : nodes->out->size;
	for (int i = 0; i < size; ++i) {
		nodes->out->data[i] = nodes->in->data[i];
	}
}
void copyBackward(Nodes_t* nodes) {
	size_t size = nodes->in->size < nodes->out->size ? nodes->in->size : nodes->out->size;
	for (int i = 0; i < size; ++i) {
		nodes->in->delta[i] = nodes->out->delta[i];
	}
}
void initNodes(Nodes_t* dst, Layer_t* before, Layer_t* after) {
	dst->structSize = sizeof(Nodes_t);
	dst->in = before;
	dst->out = after;
	dst->forward = copyForward;
	dst->backward = copyBackward;
	dst->fit = NULL;
}

/* Dense Nodes */

typedef struct DenseNodes {
	size_t structSize;
	Layer_t* in;
	Layer_t* out;
	void (*forward)(Nodes_t*);
	void (*backward)(Nodes_t*);
	void (*fit)(Nodes_t*, float l);
	float* parameter;
	size_t parameterSize;
} DenseNodes_t;
void denseForward(Nodes_t* buf) {
	if (buf->structSize != sizeof(DenseNodes_t)) {
		UninitSafememoryAndExit();
		return;
	}
	DenseNodes_t* densenodes = (DenseNodes_t*)buf;
	Layer_t* in = densenodes->in;
	Layer_t* out = densenodes->out;
	for (size_t i = 0; i < out->size; i++) {
		float* weights = densenodes->parameter + (in->size + 1) * i;
		float result = weights[in->size]; // Bias
		for (size_t j = 0; j < in->size; j++) {
			result += in->data[j] * weights[j];
		}
		out->data[i] = result;
	}
}
void denseBackward(Nodes_t* buf) {
	if (buf->structSize != sizeof(DenseNodes_t)) {
		UninitSafememoryAndExit();
		return;
	}
	DenseNodes_t* densenodes = (DenseNodes_t*)buf;
	Layer_t* in = densenodes->in;
	Layer_t* out = densenodes->out;
	for (size_t i = 0; i < in->size; i++) in->delta[i] = 0;
	for (size_t i = 0; i < out->size; i++) {
		float odi = out->delta[i];
		float* weights = densenodes->parameter + (in->size + 1) * i;
		for (size_t j = 0; j < in->size; j++) {
			in->delta[i] += odi * weights[j];
		}
	}
}
void denseFit(Nodes_t* buf, float l) {
	if (buf->structSize != sizeof(DenseNodes_t)) {
		UninitSafememoryAndExit();
		return;
	}
	DenseNodes_t* densenodes = (DenseNodes_t*)buf;
	Layer_t* in = densenodes->in;
	Layer_t* out = densenodes->out;
	for (size_t i = 0; i < out->size; i++) {
		float odi = out->delta[i];
		float* weights = densenodes->parameter + (in->size + 1) * i;
		for (size_t j = 0; j < in->size; j++) {
			weights[j] -= odi * in->data[j] * l;
		}
		weights[in->size] -= out->delta[i] * l;
	}
}
void initDenseNodes(DenseNodes_t* dst, Layer_t* in, Layer_t* out) {
	dst->structSize = sizeof(DenseNodes_t);
	dst->in = in;
	dst->out = out;
	dst->forward = denseForward;
	dst->backward = denseBackward;
	dst->fit = denseFit;
	dst->parameter = (float*)SafeMemoryAllocate((in->size + 1) * out->size * sizeof(float));
	dst->parameterSize = (in->size + 1) * out->size;
	float sigma = sqrtf(1.0f / in->size);
	for (size_t i = 0; i < out->size; i++) {
		float* weights = dst->parameter + (in->size + 1) * i;
		for (size_t j = 0; j < in->size; j++) {
			weights[j] = rand_normal(0.0f, sigma);
		}
		weights[in->size] = 0;
	}
}

/* Activation Nodes */

typedef struct ActivationNodes {
	size_t structSize;
	Layer_t* in;
	Layer_t* out;
	void (*forward)(Nodes_t*);
	void (*backward)(Nodes_t*);
	void (*fit)(Nodes_t*, float l);
	void (*func[2])(struct ActivationNodes*);
} ActivationNodes_t;
void activationForward(Nodes_t* buf) {
	if (buf->structSize != sizeof(ActivationNodes_t)) {
		UninitSafememoryAndExit();
		return;
	}
	ActivationNodes_t* activationnodes = (ActivationNodes_t*)buf;
	activationnodes->func[0](activationnodes);
}
void activationBackward(Nodes_t* buf) {
	if (buf->structSize != sizeof(ActivationNodes_t)) {
		UninitSafememoryAndExit();
		return;
	}
	ActivationNodes_t* activationnodes = (ActivationNodes_t*)buf;
	activationnodes->func[1](activationnodes);
}
void initActivationNodes(ActivationNodes_t* dst, Layer_t* in, Layer_t* out, void (*func[2])(ActivationNodes_t*)) {
	if (in->size != out->size) {
		UninitSafememoryAndExit();
		return;
	}
	dst->structSize = sizeof(ActivationNodes_t);
	dst->in = in;
	dst->out = out;
	dst->forward = activationForward;
	dst->backward = activationBackward;
	dst->fit = NULL;
	dst->func[0] = func[0];
	dst->func[1] = func[1];
}
#endif
