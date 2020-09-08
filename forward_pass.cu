#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
    static __shared__ T shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum<T>(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const int seq_len)
{
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum;

    for(int i = 0; i < seq_len; ++i)
    {
        T qk = qk_buf_[threadIdx.x + qk_offset];

        qk = __expf((float)(qk));
        float sum_val = blockReduceSum<float>(qk);

        if(threadIdx.x == 0)
        {
            s_sum = sum_val + 1e-6f;
        }
        __syncthreads();

        qk_buf_[threadIdx.x + qk_offset] = qk / (T)s_sum;

        qk_offset += seq_len;
    }
}

template <typename T>
__global__
void gradient_softmax_kernel(T* gradient_dropout, T* softmax_buf, T* dropout_mask, T dropout_prob, const int seq_len)
{
    int offset = blockIdx.x * seq_len * seq_len;

    for(int i = 0; i < seq_len; ++i)
    {
        T sum = (gradient_dropout[threadIdx.x + offset] * softmax_buf[threadIdx.x + offset]);

        float sum_val = blockReduceSum<float>(sum);

        __syncthreads();

        gradient_dropout[threadIdx.x + offset] = (gradient_dropout[threadIdx.x + offset] - (T)sum_val) * softmax_buf[threadIdx.x + offset];

        offset += seq_len;
    }
}

template <typename T>
__global__
void dropout_mul_kernel(T* dropout_rest, T* softmax_buf, T* dropout_mask, T dropout_prob)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dropout_rest[tid] = softmax_buf[tid] * dropout_mask[tid] * dropout_prob;
}

template <typename T>
__global__
void gradient_dropout_mul_kernel(T* grad_up, T* dropout_mask, T dropout_prob)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    grad_up[tid] = grad_up[tid] * dropout_mask[tid] * dropout_prob;
}

template <typename T>
__global__
void forward_fusion_kernel(T* dropout_rest, T* qk_buf_, T* dropout_mask, T dropout_prob, const int seq_len)
{
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum;

    for(int i = 0; i < seq_len; ++i)
    {
        T qk = qk_buf_[threadIdx.x + qk_offset];

        qk = __expf((float)(qk));
        float sum_val = blockReduceSum<float>(qk);

        if(threadIdx.x == 0)
        {
            s_sum = sum_val + 1e-6f;
        }
        __syncthreads();

        qk_buf_[threadIdx.x + qk_offset] = qk / (T)s_sum;

        dropout_mask[threadIdx.x + qk_offset] *= dropout_prob;

        dropout_rest[threadIdx.x + qk_offset] = qk_buf_[threadIdx.x + qk_offset] * dropout_mask[threadIdx.x + qk_offset];

        qk_offset += seq_len;
    }
}

template <typename T>
__global__
void backward_fusion_kernel(T* grad_up, T* softmax_buf, T* dropout_mask, const int seq_len)
{
    int offset = blockIdx.x * seq_len * seq_len;

    for(int i = 0; i < seq_len; ++i)
    {
        grad_up[threadIdx.x + offset] = grad_up[threadIdx.x + offset] * dropout_mask[threadIdx.x + offset];

        T sum = (grad_up[threadIdx.x + offset] * softmax_buf[threadIdx.x + offset]);

        float sum_val = blockReduceSum<float>(sum);

        __syncthreads();

        grad_up[threadIdx.x + offset] = (grad_up[threadIdx.x + offset] - (T)sum_val) * softmax_buf[threadIdx.x + offset];

        offset += seq_len;
    }
}

template<typename T>
void check(T* host_res, T* host_reference, int M, int N, bool trans, double err)
{
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N;  j++)
        {
            float v1 = 0.0;
            if(trans)
                v1 = host_reference[j * M + i];
            else
                v1 = host_reference[i * N + j];

            float v2 = host_res[i * N + j];
            // if (v1 / v2 > (1+err) || v2 / v1 > (1+err) || abs(v2 - v1) > err) {
            if (abs(v2 - v1) > err) {
                errors++;
                if (errors < 10) printf("%f %f\n", v1, v2);
            }        
        }
    }
    if (errors > 0) {
        std::cerr << "Results incorrect. Errors : " << errors << " / "<<  (M*N) << std::endl << std::endl;
    }
    else
    {
        std::cerr << "PASSED !!!" << std::endl<< std::endl;
    }
}


template<typename T>
void sparsity(T* host_res, int len)
{
    int zn = 0;
    for(int i = 0; i < len; i++)
    {
        if(host_res[i] == 0)
        {
            zn++;
        }
    }
    printf("sparsity : %f \n", (float)zn/(float)len);
}

void non_fusion(int batch_size_, int head_num_, int seq_len_)
{
    int batch_size = batch_size_;
    int head_num = head_num_;
    int seq_len = seq_len_;

    bool isCheck = false;
    if(batch_size == 1 && head_num == 16 && seq_len == 128)
        isCheck = true;

    dim3 block, grid;

    int buffer_size = batch_size * head_num * seq_len * seq_len;

    printf("----Non-fusion---- \n");
    /****************/
    /* Forward Pass */
    /****************/

    float *qk_buf_host = new float[buffer_size];
    float *qk_buf_;

    std::ifstream bin_file;
    if(isCheck)
    {
        bin_file.open("data/qk.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)qk_buf_host, buffer_size * sizeof(float));
        bin_file.close();
    }

    cudaMalloc((void**)&qk_buf_, sizeof(float) * buffer_size);
    cudaMemcpy((qk_buf_), (qk_buf_host), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    // sparsity(qk_buf_host, buffer_size);

    if(seq_len <= 32)
    block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
    else
    block.x = 1024;
    
    grid.x = batch_size * head_num;
    
    // Softmax
    softmax_kernel<float><<<grid, block>>>(qk_buf_, seq_len);


    float* softmax_buf_host = new float[buffer_size];
    float* softmax_buf_device = qk_buf_;
    cudaMemcpy((softmax_buf_host), (softmax_buf_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    // sparsity(softmax_buf_host, buffer_size);
    
    float *softmax_buf_host_bin = new float[buffer_size];


    if(isCheck)
    {
        bin_file.open("data/softmax.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)softmax_buf_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    }

    if(isCheck)
        check<float>(softmax_buf_host, softmax_buf_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-6);


    float dropout_prob;
    float *dropout_mask_host_bin = new float[buffer_size];
    float *dropout_rest_host_bin = new float[buffer_size];

    if(isCheck)
    {
        bin_file.open("data/dropout_prob.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)&dropout_prob, 4);
        bin_file.close();
        // printf("%f\n", dropout_prob);
    
        bin_file.open("data/dropout_mask.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)dropout_mask_host_bin, buffer_size * sizeof(float));
        bin_file.close();
        // sparsity(dropout_mask_host_bin, buffer_size);
    
        bin_file.open("data/dropout_rest.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)dropout_rest_host_bin, buffer_size * sizeof(float));
        bin_file.close();
        // sparsity(dropout_rest_host_bin, buffer_size);
    }

    float *dropout_rest_device;
    float *dropout_mask_device;
    cudaMalloc((void**)&dropout_rest_device, sizeof(float) * buffer_size);
    cudaMalloc((void**)&dropout_mask_device, sizeof(float) * buffer_size);
    cudaMemcpy((dropout_mask_device), (dropout_mask_host_bin), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    block.x = 256;
    grid.x = buffer_size / block.x;

    //Dropout
    dropout_mul_kernel<float><<<grid, block>>>(dropout_rest_device, softmax_buf_device, dropout_mask_device, dropout_prob);

    float *dropout_rest_host = new float[buffer_size];
    cudaMemcpy((dropout_rest_host), (dropout_rest_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    if(isCheck)
        check<float>(dropout_rest_host, dropout_rest_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-6);

    /*****************/
    /* Backward Pass */
    /*****************/

    // Dropout gradient
    float *gradient_upstream_host_bin = new float[buffer_size];
    float *gradient_dropout_host_bin = new float[buffer_size];
    float *gradient_softmax_host_bin = new float[buffer_size];

    
    if(isCheck)
    {
        bin_file.open("data/gradients_upstream.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_upstream_host_bin, buffer_size * sizeof(float));
        bin_file.close();
        // sparsity(gradient_upstream_host_bin, buffer_size);
    
        bin_file.open("data/gradients_dropout.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_dropout_host_bin, buffer_size * sizeof(float));
        bin_file.close();
        // sparsity(gradient_dropout_host_bin, buffer_size);
    
        bin_file.open("data/gradients_softmax.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_softmax_host_bin, buffer_size * sizeof(float));
        bin_file.close();
        // sparsity(gradient_softmax_host_bin, buffer_size);
    }

    float *gradient_upstream_device;
    cudaMalloc((void**)&gradient_upstream_device, sizeof(float) * buffer_size);
    cudaMemcpy((gradient_upstream_device), (gradient_upstream_host_bin), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    block.x = 256;
    grid.x = buffer_size / block.x;

    //Dropout gradient
    gradient_dropout_mul_kernel<float><<<grid, block>>>(gradient_upstream_device, dropout_mask_device, dropout_prob);

    float *gradient_dropout_host = new float[buffer_size];
    float* gradient_dropout_device = gradient_upstream_device;
    cudaMemcpy((gradient_dropout_host), (gradient_dropout_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    if(isCheck)
        check<float>(gradient_dropout_host, gradient_dropout_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-10);

    if(seq_len <= 32)
    block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
    else
    block.x = 1024;
    
    grid.x = batch_size * head_num;
    
    // Softmax gradient
    gradient_softmax_kernel<float><<<grid, block>>>(gradient_dropout_device, softmax_buf_device, dropout_mask_device, dropout_prob, seq_len);

    float *gradient_softmax_host = new float[buffer_size];
    float* gradient_softmax_device = gradient_dropout_device;
    cudaMemcpy((gradient_softmax_host), (gradient_softmax_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);


    if(isCheck)
        check<float>(gradient_softmax_host, gradient_softmax_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-10);
}

void fusion(int batch_size_, int head_num_, int seq_len_)
{
    int batch_size = batch_size_;
    int head_num = head_num_;
    int seq_len = seq_len_;

    bool isCheck = false;
    if(batch_size == 1 && head_num == 16 && seq_len == 128)
        isCheck = true;

    dim3 block, grid;

    int buffer_size = batch_size * head_num * seq_len * seq_len;

    printf("----Fusion---- \n");
    /****************/
    /* Forward Pass */
    /****************/

    float *qk_buf_host = new float[buffer_size];
    float *qk_buf_;
    float dropout_prob;
    float *dropout_mask_host_bin = new float[buffer_size];
    float *dropout_rest_host_bin = new float[buffer_size];
    float *dropout_rest_device;
    float *dropout_mask_device;

    std::ifstream bin_file;
    if(isCheck)
    {
        bin_file.open("data/qk.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)qk_buf_host, buffer_size * sizeof(float));
        bin_file.close();

        bin_file.open("data/dropout_prob.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)&dropout_prob, 4);
        bin_file.close();
    
        bin_file.open("data/dropout_mask.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)dropout_mask_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    
        bin_file.open("data/dropout_rest.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)dropout_rest_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    }
    
    cudaMalloc((void**)&dropout_rest_device, sizeof(float) * buffer_size);
    cudaMalloc((void**)&dropout_mask_device, sizeof(float) * buffer_size);
    cudaMemcpy((dropout_mask_device), (dropout_mask_host_bin), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&qk_buf_, sizeof(float) * buffer_size);
    cudaMemcpy((qk_buf_), (qk_buf_host), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    if(seq_len <= 32)
    block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
    else
    block.x = 1024;
    
    grid.x = batch_size * head_num;
    
    forward_fusion_kernel<float><<<grid, block>>>(dropout_rest_device, qk_buf_, dropout_mask_device, dropout_prob, seq_len);

    float* softmax_buf_host = new float[buffer_size];
    float* softmax_buf_device = qk_buf_;
    cudaMemcpy((softmax_buf_host), (softmax_buf_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    float *softmax_buf_host_bin = new float[buffer_size];

    if(isCheck)
    {
        bin_file.open("data/softmax.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)softmax_buf_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    }

    if(isCheck)
        check<float>(softmax_buf_host, softmax_buf_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-6);

    float *dropout_rest_host = new float[buffer_size];
    cudaMemcpy((dropout_rest_host), (dropout_rest_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    if(isCheck)
        check<float>(dropout_rest_host, dropout_rest_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-6);

    /*****************/
    /* Backward Pass */
    /*****************/

    // Dropout gradient
    float *gradient_upstream_host_bin = new float[buffer_size];
    float *gradient_dropout_host_bin = new float[buffer_size];
    float *gradient_softmax_host_bin = new float[buffer_size];

    
    if(isCheck)
    {
        bin_file.open("data/gradients_upstream.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_upstream_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    
        bin_file.open("data/gradients_dropout.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_dropout_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    
        bin_file.open("data/gradients_softmax.bin", std::ios::in | std::ios::binary);
        bin_file.read((char *)gradient_softmax_host_bin, buffer_size * sizeof(float));
        bin_file.close();
    }

    float *gradient_upstream_device;
    cudaMalloc((void**)&gradient_upstream_device, sizeof(float) * buffer_size);
    cudaMemcpy((gradient_upstream_device), (gradient_upstream_host_bin), sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

    if(seq_len <= 32)
    block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
    else
    block.x = 1024;
    
    grid.x = batch_size * head_num;
    
    backward_fusion_kernel<float><<<grid, block>>>(gradient_upstream_device, softmax_buf_device, dropout_mask_device, seq_len);

    float *gradient_softmax_host = new float[buffer_size];
    float* gradient_softmax_device = gradient_upstream_device;
    cudaMemcpy((gradient_softmax_host), (gradient_softmax_device), sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

    if(isCheck)
        check<float>(gradient_softmax_host, gradient_softmax_host_bin, batch_size*head_num, seq_len*seq_len, false, 1e-10);
}

int main()
{
    int batch_size = 32;
    int head_num = 16;
    int seq_len = 128;
    for(int i = 0; i < 10; i++)
    {
        non_fusion(batch_size, head_num, seq_len);
        fusion(batch_size, head_num, seq_len);
    }
}