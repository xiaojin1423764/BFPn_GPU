// src/utils.cuh
#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "CUFFT error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSPARSE error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)

// 设备内存管理类
template<typename T>
class DeviceArray {
private:
    T* d_ptr;
    size_t size;
    bool owned;

public:
    DeviceArray() : d_ptr(nullptr), size(0), owned(false) {}
    
    explicit DeviceArray(size_t n) : size(n), owned(true) {
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
    }
    
    DeviceArray(T* ptr, size_t n) : d_ptr(ptr), size(n), owned(false) {}
    
    ~DeviceArray() {
        if (owned && d_ptr) cudaFree(d_ptr);
    }
    
    // 禁止拷贝
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    
    // 允许移动
    DeviceArray(DeviceArray&& other) noexcept 
        : d_ptr(other.d_ptr), size(other.size), owned(other.owned) {
        other.d_ptr = nullptr;
        other.size = 0;
        other.owned = false;
    }
    
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (owned && d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            size = other.size;
            owned = other.owned;
            other.d_ptr = nullptr;
            other.size = 0;
            other.owned = false;
        }
        return *this;
    }
    
    void allocate(size_t n) {
        if (owned && d_ptr) cudaFree(d_ptr);
        size = n;
        owned = true;
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
    }
    
    void free() {
        if (owned && d_ptr) {
            cudaFree(d_ptr);
            d_ptr = nullptr;
            size = 0;
        }
    }
    
    void copyFromHost(const T* h_ptr, size_t n = 0) {
        if (n == 0) n = size;
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copyToHost(T* h_ptr, size_t n = 0) const {
        if (n == 0) n = size;
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void copyFromDevice(const T* src, size_t n = 0) {
        if (n == 0) n = size;
        CUDA_CHECK(cudaMemcpy(d_ptr, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    
    void setZero() {
        if (d_ptr && size > 0) {
            CUDA_CHECK(cudaMemset(d_ptr, 0, size * sizeof(T)));
        }
    }
    
    T* data() const { return d_ptr; }
    size_t getSize() const { return size; }
    
    T& operator[](size_t i) { return d_ptr[i]; }
    const T& operator[](size_t i) const { return d_ptr[i]; }
};

//  pitched 2D数组
template<typename T>
class PitchedArray {
private:
    T* d_ptr;
    size_t width, height;
    size_t pitch;
    bool owned;

public:
    PitchedArray() : d_ptr(nullptr), width(0), height(0), pitch(0), owned(false) {}
    
    PitchedArray(size_t w, size_t h) : width(w), height(h), owned(true) {
        CUDA_CHECK(cudaMallocPitch(&d_ptr, &pitch, w * sizeof(T), h));
    }
    
    ~PitchedArray() {
        if (owned && d_ptr) cudaFree(d_ptr);
    }
    
    void allocate(size_t w, size_t h) {
        if (owned && d_ptr) cudaFree(d_ptr);
        width = w;
        height = h;
        owned = true;
        CUDA_CHECK(cudaMallocPitch(&d_ptr, &pitch, w * sizeof(T), h));
    }
    
    size_t getPitch() const { return pitch; }
    T* data() const { return d_ptr; }
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
};

#endif
