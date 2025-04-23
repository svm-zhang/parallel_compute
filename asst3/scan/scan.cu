#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
upsweep_kernel(int N, int d, int *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = 2 * d;
  int idx = index * offset;
  if (idx + offset - 1 < N) {
    result[idx+offset-1] += result[idx+d-1];
  }
}

__global__ void
downsweep_kernel(int N, int d, int *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = 2 * d;
  int idx = index * offset;
  if (idx + offset - 1 < N) {
    int tmp = result[idx+d-1];
    result[idx+d-1] = result[idx+offset-1];
    result[idx+offset-1] += tmp;
  }
}


// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    for (int d = 1; d < N/2; d *= 2) {
      // one thread allocated per iteration
      // inner loop has N / 2*d iterations per iteration of outer loop
      int iters = N / (2*d);
      // limit the number of threads to launch when THREADS_PER_BLOCK > iters
      int threadsPerBlock = std::min(iters, THREADS_PER_BLOCK);
      int blocks = (iters + threadsPerBlock - 1) / threadsPerBlock;

      upsweep_kernel<<<blocks, threadsPerBlock>>>(N, d, result);
      // need to wait for all threads finished before moving to the next depth
      cudaDeviceSynchronize();
    }

    // set the last element to be zero
    int zero = 0;
    cudaMemcpy(&result[N-1], &zero, sizeof(int), cudaMemcpyHostToDevice);

    for (int d = N/2; d >= 1; d /= 2 ) {
      int iters = N / (2*d);
      int threadsPerBlock = std::min(iters, THREADS_PER_BLOCK);
      int blocks = (iters + threadsPerBlock - 1) / threadsPerBlock;
      // printf("iters=%d\ttpb=%d\tblocks=%d\n", iters, threadsPerBlock, blocks);

      downsweep_kernel<<<blocks, threadsPerBlock>>>(N, d, result);
      cudaDeviceSynchronize();
    }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void
compare_adjacent(int length, int *input, int *mask) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  mask[index] = 0;
  // compare current with the next
  if (index < length - 1 && input[index] == input[index+1]) {
      mask[index] = 1;
  }
}


__global__ void
scatter(int length, int *mask, int *scan, int *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // mask[i] = 1 then scan[i] = s the result[s] = i
  result[index] = 0;
  if (index < length - 1) {
    if (mask[index] == 1) {
      result[scan[index]] = index;
    }
  }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int count = 0;

    // get a mask array where mask[i] = 1 if input[i] == input[i+1] 
    int *device_mask = nullptr;
    cudaMalloc((void **)&device_mask, length * sizeof(int));
    // no need to launch so many threads
    int threadsPerBlock = std::min(length, THREADS_PER_BLOCK);
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    compare_adjacent<<<blocks, threadsPerBlock>>>(length, device_input, device_mask);

    int *device_scan = nullptr;
    cudaMalloc((void **)&device_scan, length * sizeof(int));
    cudaMemcpy(device_scan, device_mask, length*sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan(device_mask, length, device_scan);

    // the last element of exclusive_scan tells you the number of repeats
    cudaMemcpy(&count, &device_scan[length-1], sizeof(int), cudaMemcpyDeviceToHost);

    scatter<<<blocks, threadsPerBlock>>>(length, device_mask, device_scan, device_output);

    /* Debug purpose
    int* input = new int[length];
    cudaMemcpy(input, device_input, length*sizeof(int), cudaMemcpyDeviceToHost);
    int* mask_result = new int[length];
    cudaMemcpy(mask_result, device_mask, length*sizeof(int), cudaMemcpyDeviceToHost);
    int* scan_result = new int[length];
    cudaMemcpy(scan_result, device_scan, length*sizeof(int), cudaMemcpyDeviceToHost);
    int* scatter_result = new int[length];
    cudaMemcpy(scatter_result, device_output, length*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++) {
      printf("input[i]=%d\tmask[i]=%d\tscan[i]=%d\tscatter[i]=%d\n", input[i], mask_result[i], scan_result[i], scatter_result[i]);
    }
    */

    return count; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
