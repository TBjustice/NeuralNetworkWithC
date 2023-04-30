// SafeMemory
// Created on April 29, 2023 by TrueBlueFeather
// MIT License

#ifndef SAFEMEMORY
#define SAFEMEMORY

#ifdef SHOWMEMORYLOG
#include <stdio.h>
#endif

#include <stdlib.h>

typedef struct {
	void* before;
	void* after;
} SafeMemoryHeader;
SafeMemoryHeader safememorystart;
SafeMemoryHeader* safememoryend = &safememorystart;
void InitSafeMemory() {
	safememorystart.before = NULL;
	safememorystart.after = NULL;
}
void UninitSafememory() {
	while (safememoryend != &safememorystart) {
		void* buf2 = (SafeMemoryHeader*)safememoryend->before;
#ifdef SHOWMEMORYLOG
		printf("Free[%llx]\n", (unsigned long long)safememoryend);
#endif
		free(safememoryend);
		safememoryend = (SafeMemoryHeader*)buf2;
	}
}
void UninitSafememoryAndExit() {
	UninitSafememory();
	exit(1);
}
void* SafeMemoryAllocate(size_t size) {
	void* buf1 = malloc(size + sizeof(SafeMemoryHeader));
	void* buf2 = (void*)safememoryend;
	if (buf1 == NULL) {
#ifdef SHOWMEMORYLOG
		printf("SafeMemoryAllocate Fail\n");
#endif
		UninitSafememoryAndExit();
		return NULL;
	}
	safememoryend->after = buf1;
	safememoryend = (SafeMemoryHeader*)buf1;
	safememoryend->before = buf2;
	safememoryend->after = NULL;
#ifdef SHOWMEMORYLOG
	printf("Allocate[%llx]\n", (unsigned long long)safememoryend);
#endif
	return (void*)(safememoryend + 1);
}
void SafeMemoryRelease(void* data) {
	SafeMemoryHeader* buf = (SafeMemoryHeader*)data - 1;
	if (buf->after == NULL) safememoryend = (SafeMemoryHeader*)buf->before;
	else {
		((SafeMemoryHeader*)buf->after)->before = (SafeMemoryHeader*)buf->before;
		((SafeMemoryHeader*)buf->before)->after = (SafeMemoryHeader*)buf->after;
	}
	free(buf);
#ifdef SHOWMEMORYLOG
	printf("Free[%llx]\n", (unsigned long long)buf);
#endif
}
#endif
