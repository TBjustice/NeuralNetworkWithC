#ifndef CNEURAL_ACTIVATION_FUNCTIONS
#define CNEURAL_ACTIVATION_FUNCTIONS

#include "Layer.h"
#include <math.h>

void F_Linear(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->in->data;
	float* dst = activationnodes->out->data;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		*dst = *src;
		++dst;
		++src;
	}
}
void DF_Linear(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->out->delta;
	float* dst = activationnodes->in->delta;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		*dst = *src;
		++dst;
		++src;
	}

}
void (*Linear[2])(ActivationNodes_t*) = { F_Linear, DF_Linear };

void F_ReLU(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->in->data;
	float* dst = activationnodes->out->data;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		*dst = *src < 0 ? 0 : *src;
		++dst;
		++src;
	}
}
void DF_ReLU(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->out->delta;
	float* dst = activationnodes->in->delta;
	float* data = activationnodes->in->data;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		*dst = *data < 0 ? 0 : *src;
		++dst;
		++data;
		++src;
	}
}
void (*ReLU[2])(ActivationNodes_t*) = { F_ReLU, DF_ReLU };

void F_Softmax(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->in->data;
	float* dst = activationnodes->out->data;
	float C = -1000000.0f;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		C = C < src[i] ? src[i] : C;
	}
	float sum = 0;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		sum += expf(src[i] - C);
	}
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		dst[i] = expf(src[i] - C) / sum;
	}
}
void DF_Softmax(ActivationNodes_t* activationnodes) {
	float* src = activationnodes->out->delta;
	float* dst = activationnodes->in->delta;
	float* data = activationnodes->out->data;
	for (size_t i = 0; i < activationnodes->in->size; ++i) {
		float result = 0;
		for (size_t j = 0; j < activationnodes->in->size; ++j) {
			result += (data[j] * ((i == j ? 1 : 0) - data[i])) * src[j];
		}
		dst[i] = result;
	}
}
void (*Softmax[2])(ActivationNodes_t*) = { F_Softmax, DF_Softmax };
#endif
