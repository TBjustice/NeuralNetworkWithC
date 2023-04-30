#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "safememory.h"
#include "Layer.h"
#include "ActivationFunctions.h"
#include "MNISTUtil.h"

int main(void) {
	InitSafeMemory();

	srand(0);

	MNISTIMAGE_t mnistImage = { 0 };
	MNISTLABEL_t mnistLabel = { 0 };

	FILE* fp = fopen("train-images.idx3-ubyte", "rb");
	readMNISTImage(&mnistImage, fp);
	fclose(fp);

	fp = fopen("train-labels.idx1-ubyte", "rb");
	readMNISTLabel(&mnistLabel, fp);
	fclose(fp);

	if (mnistLabel.labels == NULL) {
		UninitSafememoryAndExit();
		return 1;
	}

	Layer_t layers[9];
	DenseNodes_t dense1, dense2, dense3, dense4;
	ActivationNodes_t activation1, activation2, activation3, activation4;
	Nodes_t* nodes[8] = {
		(Nodes_t*)&dense1, (Nodes_t*)&activation1,
		(Nodes_t*)&dense2, (Nodes_t*)&activation2,
		(Nodes_t*)&dense3, (Nodes_t*)&activation3,
		(Nodes_t*)&dense4, (Nodes_t*)&activation4,
	};

	initLayer(&layers[0], mnistImage.nrow * mnistImage.ncol);

	initLayer(&layers[1], 512);
	initLayer(&layers[2], 512);

	initLayer(&layers[3], 512);
	initLayer(&layers[4], 512);

	initLayer(&layers[5], 256);
	initLayer(&layers[6], 256);

	initLayer(&layers[7], 10);
	initLayer(&layers[8], 10);

	for (int i = 0; i < 4; i++) {
		initDenseNodes((DenseNodes_t*)nodes[i * 2], &layers[i * 2], &layers[i * 2 + 1]);
		if (i != 3)initActivationNodes((ActivationNodes_t*)nodes[i * 2 + 1], &layers[i * 2 + 1], &layers[i * 2 + 2], Linear);
		else initActivationNodes((ActivationNodes_t*)nodes[i * 2 + 1], &layers[i * 2 + 1], &layers[i * 2 + 2], Softmax);
	}

	for (size_t iteration = 0; iteration < 60; iteration++) {
		size_t correctCount = 0;

		for (size_t batch = 0; batch < 300; batch++) {
			size_t target = rand() % mnistImage.nimages;

			MNISTImage2Float(layers[0].data, &mnistImage, target);

			for (int i = 0; i < 8; i++) {
				nodes[i]->forward(nodes[i]);
			}


			float maxvalue = 0;
			size_t maxindex = 0;
			for (int i = 0; i < 10; i++) {
				if (layers[8].data[i] > maxvalue) {
					maxvalue = layers[8].data[i];
					maxindex = i;
				}
			}
			if (maxindex == mnistLabel.labels[target]) {
				correctCount++;
			}

			float loss = 0.0f;
			for (int i = 0; i < 10; i++) {
				float dif = layers[8].data[i] - (mnistLabel.labels[target] == i ? 1 : 0);
				loss += 0.5f * dif * dif;
			}
			for (int i = 0; i < 10; i++) {
				float dif = layers[8].data[i] - (mnistLabel.labels[target] == i ? 1 : 0);
				layers[8].delta[i] = loss * dif;
			}
			for (int i = 7; i >= 0; i--) {
				nodes[i]->backward(nodes[i]);
			}

			float learn = iteration < 30 ? 0.01f : 0.002f;
			for (int i = 0; i < 8; i++) {
				if (nodes[i]->fit)nodes[i]->fit(nodes[i], learn);
			}
		}
		printf("Grade : %lld/300\tAccuracy : %f\n", correctCount, (float)correctCount / 300.0f);
	}

	for (int i = 0; i < 20; i++) {
		size_t target = rand() % mnistImage.nimages;
		showMNISTImage(&mnistImage, target);
		printf("Answer = %d\n", mnistLabel.labels[target]);

		MNISTImage2Float(layers[0].data, &mnistImage, target);
		for (int i = 0; i < 8; i++) {
			nodes[i]->forward(nodes[i]);
		}
		float maxvalue = 0;
		size_t maxindex = 0;
		for (int i = 0; i < 10; i++) {
			printf("%ld[%f] ", i, layers[8].data[i]);
			if (layers[8].data[i] > maxvalue) {
				maxvalue = layers[8].data[i];
				maxindex = i;
			}
		}
		printf("\nExpectation = %lld\n\n\n", maxindex);

	}

	UninitSafememory();
}
