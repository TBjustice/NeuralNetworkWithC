#ifndef MNISTUTIL
#define MNISTUTIL

#include <stdio.h>
#include <stdint.h>
#include "safememory.h"

typedef struct MNISTIMAGE {
	uint32_t magic;
	uint32_t nimages;
	uint32_t nrow;
	uint32_t ncol;
	uint8_t* images;
} MNISTIMAGE_t;

void readuint32(uint32_t* dst, FILE* fp) {
	unsigned char buf[4];
	fread(buf, sizeof(unsigned char), 4, fp);
	*dst = 0;
	for (int i = 0; i < 4; i++) {
		*dst <<= 8;
		*dst |= buf[i];
	}
}

void readMNISTImage(MNISTIMAGE_t* mnistImage, FILE* fp) {
	readuint32(&mnistImage->magic, fp);
	readuint32(&mnistImage->nimages, fp);
	readuint32(&mnistImage->nrow, fp);
	readuint32(&mnistImage->ncol, fp);
	mnistImage->images = (uint8_t*)SafeMemoryAllocate(mnistImage->nimages * mnistImage->nrow * mnistImage->ncol);
	fread(mnistImage->images, sizeof(unsigned char), mnistImage->nimages * mnistImage->nrow * mnistImage->ncol, fp);
}

void showMNISTImage(MNISTIMAGE_t* mnistImage, size_t idx) {
	uint8_t* image = &mnistImage->images[mnistImage->nrow * mnistImage->ncol * idx];
	for (int y = 0; y < mnistImage->nrow; y++) {
		for (int x = 0; x < mnistImage->ncol; x++) {
			if (image[y * mnistImage->ncol + x] > 180) printf("#");
			else if (image[y * mnistImage->ncol + x] > 128) printf("H");
			else printf(" ");
		}
		printf("\n");
	}
}
void MNISTImage2Float(float* dst, MNISTIMAGE_t* mnistImage, size_t idx) {
	uint8_t* image = &mnistImage->images[mnistImage->nrow * mnistImage->ncol * idx];
	for (size_t i = 0; i < mnistImage->nrow * mnistImage->ncol; i++) {
		dst[i] = (float)image[i] / 256;
	}
}

typedef struct MNISTLABEL {
	uint32_t magic;
	uint32_t nimages;
	uint8_t* labels;
} MNISTLABEL_t;

void readMNISTLabel(MNISTLABEL_t* mnistLabel, FILE* fp) {
	readuint32(&mnistLabel->magic, fp);
	readuint32(&mnistLabel->nimages, fp);
	mnistLabel->labels = (uint8_t*)SafeMemoryAllocate(mnistLabel->nimages);
	fread(mnistLabel->labels, sizeof(unsigned char), mnistLabel->nimages, fp);
}

#endif
