#include <ctime>

#include <iostream>
#include <iomanip>

#include <assert.h>

#include <cuda.h>

// inner product based matmul
template <typename val_t, typename pos_t>
__global__ void K0(const val_t *A, const val_t *B, val_t *C,
                   const pos_t M, const pos_t K, const pos_t N)
{
    for (pos_t m = blockIdx.x * blockDim.x + threadIdx.x; m < M; m += gridDim.x * blockDim.x)
    {
        for (pos_t n = blockIdx.y * blockDim.y + threadIdx.y; n < N; n += gridDim.y * blockDim.y)
        {
            val_t accum = 0;
            for (pos_t k = 0; k < K; k++)
            {
                accum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = accum;
        }
    }
}

template <typename val_t, typename pos_t>
inline void K0_launcher(const val_t *A, const val_t *B, val_t *C,
                        const pos_t M, const pos_t K, const pos_t N)
{
    dim3 gd(1, 1, 1);
    dim3 bd(32, 32, 1);
    K0<<<gd, bd>>>(A, B, C, M, K, N);
}

// https://github.com/BryanCatanzaro/inplace
// sth wrong as in below, inplace transpose harder than I expected
template <typename val_t, typename pos_t, pos_t TM, pos_t TN>
__global__ void ITK(val_t *mat, const pos_t M, const pos_t N)
{
    assert(blockDim.x <= TM);
    assert(blockDim.y <= TN);
    // assert(blockDim.x == blockDim.y);
    __shared__ val_t tile[2][TM][TN];
    // bank conflicit as it will access elems by rows and cols, cache on registers?
    for (pos_t m = blockIdx.x * blockDim.x + threadIdx.x; m < M; m += gridDim.x * blockDim.x)
    {
        for (pos_t n = blockIdx.y * blockDim.y + threadIdx.y; n < N; n += gridDim.y * blockDim.y)
        {
            if ((m > n) && (M >= N) || (m < n) && (M < N))
            {
                tile[0][n % blockDim.y][m % blockDim.x] = mat[m * N + n];
                tile[1][m % blockDim.x][n % blockDim.y] = mat[n * M + m];
            }
            __syncthreads();
            if ((m > n) && (M >= N) || (m < n) && (M < N))
            {
                mat[n * M + m] = tile[0][m % blockDim.x][n % blockDim.y];
                mat[m * N + n] = tile[1][m % blockDim.x][n % blockDim.y];
            }
        }
    }
}

template <typename val_t, typename pos_t, pos_t TM, pos_t TN>
__global__ void TK(val_t *original, const pos_t M, const pos_t N, val_t *transposed)
{
    assert(blockDim.x <= TM);
    assert(blockDim.y <= TN);
    __shared__ val_t tile[TM][TN];
    // bank conflicit as it will access elems by rows and cols, cache on registers?
    for (pos_t m = blockIdx.x * blockDim.x + threadIdx.x; m < M; m += gridDim.x * blockDim.x)
    {
        for (pos_t n = blockIdx.y * blockDim.y + threadIdx.y; n < N; n += gridDim.y * blockDim.y)
        {
            tile[m % blockDim.x][n % blockDim.y] = original[m * N + n];
            __syncthreads();
            transposed[n * M + m] = tile[m % blockDim.x][n % blockDim.y];
        }
    }
}

template <typename val_t, typename pos_t>
void transpose(val_t *original,  const pos_t nr, const pos_t nc, bool inplace=true, val_t *transposed=nullptr)
{
    dim3 gd(6, 8);
    dim3 bd(32, 32);
    assert(inplace == false); // as not yet finished
    // ITK<val_t, pos_t, 33, 33><<<gd, bd>>>(original, nr, nc);

    TK<val_t, pos_t, 33, 33><<<gd, bd>>>(original, nr, nc, transposed);
}

// outer product based matmul
template <typename val_t, typename pos_t, pos_t MAX_M, pos_t MAX_N>
__global__ void K1(const val_t *A, const val_t *B, val_t *C,
                   const pos_t M, const pos_t K, const pos_t N)
{
    __shared__ val_t AC[MAX_M], BR[MAX_N]; // A column, B row, A better transposed but not yet
    assert(M <= MAX_M); // get to re-balance MAX_M and MAX_N if one of them too large
    assert(N <= MAX_N); // get to add another 2D loop if both MAX_M + MAX_N too large
    for (int k = blockIdx.x; k < K; k += gridDim.x)
    {
        for (pos_t m = threadIdx.x; m < M; m += blockDim.x)
        {
            AC[m] = A[k * M + m]; // if A is column-major
            // AC[m] = A[m * K + k]; // if A is row-major
        }

        for (pos_t n = threadIdx.x; n < N; n += blockDim.x)
        {
            BR[n] = B[k * N + n];
        }

        __syncthreads();

        for (pos_t i = threadIdx.x; i < M * N; i += blockDim.x)
        {
            pos_t m = i / N, n = i % N;
            val_t v = AC[m] *  BR[n];
            atomicAdd(C + m * N  + n, v);
        }
    }
}

template <typename val_t, typename pos_t>
inline void K1_launcher(val_t *A, const val_t *B, val_t *C,
                        const pos_t M, const pos_t K, const pos_t N)
{
    cudaMemset(C, 0, M * N * sizeof(val_t)); // w/o memset also w/ correct res?
    K1<val_t, pos_t, 5000, 5000><<<K, 1024>>>(A, B, C, M, K, N);
}

#include <stdint.h>

#define BS 128
#define KS 16

// vanilla tile based matmul, recursively divide and conquer, too many loops
__global__ void K2(const int *A, const int *B, int *C,
                   const uint32_t M, const uint32_t K, const uint32_t N)
{
    for (uint32_t m0 = blockIdx.x * BS; m0 < M; m0 += gridDim.x * BS)
    {
        for (uint32_t n0 = blockIdx.y * BS; n0 < N; n0 += gridDim.y * BS)
        {
            // register int4 CT[2][2][4] = {make_int4(0, 0, 0, 0)};
            register int CT[2][2][4][4] = {0};

            // how to share shm content cross blocks?
            for (uint32_t k0 = 0; k0 < K; k0 += KS) // KS = 16 = blockDim.x .y
            {
                __shared__ int AT[KS][BS];

                for (uint32_t i = threadIdx.x; i < BS; i += blockDim.x)
                {
                    for (uint32_t j = threadIdx.y; j < KS; j += blockDim.y)
                    {
                        uint32_t m = m0 + i, k = k0 + j;
                        if (m < M && k < K)
                        {
                            AT[j][i] = A[m * K + k];
                        }
                        else
                        {
                            AT[j][i] = 0;
                        }
                    }
                }

                __shared__ int BT[KS][BS];

                for (uint32_t i = threadIdx.x; i < KS; i += blockDim.x)
                {
                    for (uint32_t j = threadIdx.y; j < BS; j += blockDim.y)
                    {
                        uint32_t k = k0 + i, n = n0 + j;
                        if (k < K && n < N)
                        {
                            BT[i][j] = B[k * N + n];
                        }
                        else
                        {
                            BT[i][j] = 0;
                        }
                    }
                }

                register int4 AC[2], BR[2];

                #pragma unroll
                for (int tk = 0; tk < KS; tk++)
                {
                    // will make_uint4 trigger some smart compiler behaviour to avoid explicit use of int4 dtype here?
                    AC[0] = make_int4(AT[tk][threadIdx.x*4], AT[tk][threadIdx.x*4+1], AT[tk][threadIdx.x*4+2], AT[tk][threadIdx.x*4+3]);
                    AC[1] = make_int4(AT[tk][(blockDim.x+threadIdx.x)*4], AT[tk][(blockDim.x+threadIdx.x)*4+1], AT[tk][(blockDim.x+threadIdx.x)*4+2], AT[tk][(blockDim.x+threadIdx.x)*4+3]);
                    BR[0] = make_int4(BT[tk][threadIdx.y*4], BT[tk][threadIdx.y*4+1], BT[tk][threadIdx.y*4+2], BT[tk][threadIdx.y*4+3]);
                    BR[1] = make_int4(BT[tk][(blockDim.y+threadIdx.y)*4], BT[tk][(blockDim.y+threadIdx.y)*4+1], BT[tk][(blockDim.y+threadIdx.y)*4+2], BT[tk][(blockDim.y+threadIdx.y)*4+3]);


                    CT[0][0][0][0] += AC[0].x * BR[0].x;
                    CT[0][0][0][1] += AC[0].x * BR[0].y;
                    CT[0][0][0][2] += AC[0].x * BR[0].z;
                    CT[0][0][0][3] += AC[0].x * BR[0].w;
                    CT[0][0][1][0] += AC[0].y * BR[0].x;
                    CT[0][0][1][1] += AC[0].y * BR[0].y;
                    CT[0][0][1][2] += AC[0].y * BR[0].z;
                    CT[0][0][1][3] += AC[0].y * BR[0].w;
                    CT[0][0][2][0] += AC[0].z * BR[0].x;
                    CT[0][0][2][1] += AC[0].z * BR[0].y;
                    CT[0][0][2][2] += AC[0].z * BR[0].z;
                    CT[0][0][2][3] += AC[0].z * BR[0].w;
                    CT[0][0][3][0] += AC[0].w * BR[0].x;
                    CT[0][0][3][1] += AC[0].w * BR[0].y;
                    CT[0][0][3][2] += AC[0].w * BR[0].z;
                    CT[0][0][3][3] += AC[0].w * BR[0].w;

                    CT[0][1][0][0] += AC[0].x * BR[1].x;
                    CT[0][1][0][1] += AC[0].x * BR[1].y;
                    CT[0][1][0][2] += AC[0].x * BR[1].z;
                    CT[0][1][0][3] += AC[0].x * BR[1].w;
                    CT[0][1][1][0] += AC[0].y * BR[1].x;
                    CT[0][1][1][1] += AC[0].y * BR[1].y;
                    CT[0][1][1][2] += AC[0].y * BR[1].z;
                    CT[0][1][1][3] += AC[0].y * BR[1].w;
                    CT[0][1][2][0] += AC[0].z * BR[1].x;
                    CT[0][1][2][1] += AC[0].z * BR[1].y;
                    CT[0][1][2][2] += AC[0].z * BR[1].z;
                    CT[0][1][2][3] += AC[0].z * BR[1].w;
                    CT[0][1][3][0] += AC[0].w * BR[1].x;
                    CT[0][1][3][1] += AC[0].w * BR[1].y;
                    CT[0][1][3][2] += AC[0].w * BR[1].z;
                    CT[0][1][3][3] += AC[0].w * BR[1].w;

                    CT[1][0][0][0] += AC[1].x * BR[0].x;
                    CT[1][0][0][1] += AC[1].x * BR[0].y;
                    CT[1][0][0][2] += AC[1].x * BR[0].z;
                    CT[1][0][0][3] += AC[1].x * BR[0].w;
                    CT[1][0][1][0] += AC[1].y * BR[0].x;
                    CT[1][0][1][1] += AC[1].y * BR[0].y;
                    CT[1][0][1][2] += AC[1].y * BR[0].z;
                    CT[1][0][1][3] += AC[1].y * BR[0].w;
                    CT[1][0][2][0] += AC[1].z * BR[0].x;
                    CT[1][0][2][1] += AC[1].z * BR[0].y;
                    CT[1][0][2][2] += AC[1].z * BR[0].z;
                    CT[1][0][2][3] += AC[1].z * BR[0].w;
                    CT[1][0][3][0] += AC[1].w * BR[0].x;
                    CT[1][0][3][1] += AC[1].w * BR[0].y;
                    CT[1][0][3][2] += AC[1].w * BR[0].z;
                    CT[1][0][3][3] += AC[1].w * BR[0].w;

                    CT[1][1][0][0] += AC[1].x * BR[1].x;
                    CT[1][1][0][1] += AC[1].x * BR[1].y;
                    CT[1][1][0][2] += AC[1].x * BR[1].z;
                    CT[1][1][0][3] += AC[1].x * BR[1].w;
                    CT[1][1][1][0] += AC[1].y * BR[1].x;
                    CT[1][1][1][1] += AC[1].y * BR[1].y;
                    CT[1][1][1][2] += AC[1].y * BR[1].z;
                    CT[1][1][1][3] += AC[1].y * BR[1].w;
                    CT[1][1][2][0] += AC[1].z * BR[1].x;
                    CT[1][1][2][1] += AC[1].z * BR[1].y;
                    CT[1][1][2][2] += AC[1].z * BR[1].z;
                    CT[1][1][2][3] += AC[1].z * BR[1].w;
                    CT[1][1][3][0] += AC[1].w * BR[1].x;
                    CT[1][1][3][1] += AC[1].w * BR[1].y;
                    CT[1][1][3][2] += AC[1].w * BR[1].z;
                    CT[1][1][3][3] += AC[1].w * BR[1].w;
                }
            }

            // if ((m0 + threadIdx.x * 4) < M && n0 + threadIdx.y * 4 < N)
            //     ((int4 *)C)[((m0 + threadIdx.x * 4) * N + n0 + threadIdx.y * 4)] = CT[0][0][0];
            // ((int4 *)C)[0] = CT[0][0][0]; // this line works for write mockup, but the above failed

            #pragma unroll // so backoff to w/o packed int4 write
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        for (int v = 0; v < 4; v++)
                        {
                            uint32_t m = m0 + threadIdx.x * 4 + i * blockDim.x * 4 + k;
                            uint32_t n = n0 + threadIdx.y * 4 + j * blockDim.y * 4 + v;
                            if (m < M && n < N)
                            {
                                C[m * N + n] = CT[i][j][k][v];
                            }
                        }
                    }
                }
            }
            // 4 vals packed is kind of troublesome in programming
        }
    }
}

inline void K2_launcher(const int *A, const int *B, int *C,
                        const uint32_t M, const uint32_t K, const uint32_t N)
{
    dim3 gd(6 * 3, 8 * 2, 1);
    dim3 bd(16, 16, 1); // square tile? more threads? shared mem for larger block?
    K2<<<gd, bd>>>(A, B, C, M, K, N);
}

// tall and thin matmul

// tiny matmul

// for ffn?

// loop over a wide range of M, N, K, dtype, CC/TC combinations

// okay...cublas

// what if M, N, K or grid size or block size out of boundary?

// what if we want to decouple the building blocks to variants

// okay...cutlass

template <typename val_t, typename pos_t>
struct Matmul
{
    val_t *A, *B, *C, *refC;
    val_t *dA, *dB, *dC;
    val_t *TA, *dTA;
    pos_t M, K, N;

    cudaEvent_t es, ee;
    float t_ms;

    void host_alloc()
    {
        A = (val_t *)malloc(M * K * sizeof(val_t));
        B = (val_t *)malloc(K * N * sizeof(val_t));
        C = (val_t *)malloc(M * N * sizeof(val_t));
        refC = (val_t *)malloc(M * N * sizeof(val_t));
        TA = (val_t *)malloc(M * K * sizeof(val_t));
    }

    void host_free()
    {
        free(A);
        free(B);
        free(C);
        free(refC);
        free(TA);
        A = B = C = refC = TA = nullptr;
    }

    void fill_vals()
    {
        std::srand(std::time({}));
        for (pos_t m = 0; m < M; m++)
        {
            for (pos_t k = 0; k < K; k++)
            {
                A[m * K + k] = std::rand() % 2333;
            }
        }

        for (pos_t k = 0; k < K; k++)
        {
            for (pos_t n = 0; n < N; n++)
            {
                B[k * N + n] = std::rand() % 2333;
            }
        }
    }


    void get_ref()
    {
        for (pos_t m = 0; m < M; m++)
        {
            for (pos_t n = 0; n < N; n++)
            {
                val_t val = 0;
                for (pos_t k = 0; k < K; k++)
                {
                    val += A[m * K + k] * B[k * N + n];
                }
                refC[m * N + n] = val;
            }
        }
    }


    void device_alloc()
    {
        cudaMalloc(&dA, M * K * sizeof(val_t));
        cudaMalloc(&dB, K * N * sizeof(val_t));
        cudaMalloc(&dC, M * N * sizeof(val_t));

        cudaMalloc(&dTA, M * K * sizeof(val_t));
    }


    void device_free()
    {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaFree(dTA);

        dA = dB = dC = dTA = nullptr;
    }


    void h2d()
    {
        cudaMemcpy(dA, A, M * K * sizeof(val_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, K * N * sizeof(val_t), cudaMemcpyHostToDevice);
    }


    inline void tic(std::string msg)
    {
        std::cout << msg << " ";
        cudaEventRecord(es);
    }


    inline void toc()
    {
        cudaEventRecord(ee);
        cudaEventSynchronize(ee);
        cudaEventElapsedTime(&t_ms, es, ee);
        std::cout << "elapsed time: " << t_ms << "ms" << std::endl;
    }

    inline void get_res()
    {
        // K0_launcher(dA, dB, dC, M, K, N);
        // K1_launcher(dTA, dB, dC, M, K, N);
        K2_launcher(dA, dB, dC, M, K, N);
    }

    void d2h()
    {
        cudaMemcpy(C, dC, M * N * sizeof(val_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(TA, dTA, M * K * sizeof(val_t), cudaMemcpyDeviceToHost);
    }

    void print_mat(std::string mat_name, val_t *data, pos_t nr, pos_t nc)
    {
        std::cout << mat_name << std::endl; // << std::setw(16);
        for (pos_t ir = 0; ir < nr; ir++)
        {
            for (pos_t ic = 0; ic < nc; ic++)
            {
                std::cout << data[ir * nc + ic] << "\t";
            }
            std::cout << std::endl;
        }
    }

    void print()
    {
        print_mat("A", A, M, K);
        print_mat("B", B, K, N);
        print_mat("refC", refC, M, N);
        print_mat("C", C, M, N);
    }

    bool match()
    {
        bool matched = true;
        for (pos_t i = 0; i < M * N; i++)
        {
            if (C[i] != refC[i])
            {
                std::cout << "mismatch, pos in 1D array: " << i << std::endl;
                matched = false;
                break;
            }
        }
        return matched;
    }

    void compute()
    {
        tic("gpu matmul");
        get_res();
        toc();
        d2h();
        assert(match());
        // std::cout << "correct? " << (match()?"yes":"no") << std::endl;
    }

    Matmul() = delete;

    Matmul(pos_t m, pos_t k, pos_t n) : M(m), K(k), N(n)
    {
        host_alloc();
        fill_vals();
        get_ref();

        device_alloc();
        cudaEventCreate(&es);
        cudaEventCreate(&ee);

        h2d();
        transpose(dA, M, K, false, dTA);
    }


    ~Matmul()
    {
        host_free();
        device_free();
        cudaEventDestroy(es);
        cudaEventDestroy(ee);
    }
};

__global__ void fooKernel()
{
}

template <typename val_t, typename pos_t>
void transpose_test(Matmul<val_t, pos_t> &mm)
{
    // mm.print_mat("original A", mm.A, 51, 31);
    // mm.d2h();
    // mm.print_mat("transposed A", mm.TA, 31, 51);
    // mm.print();
}

int main()
{
    fooKernel<<<1, 1>>>();

    // Matmul<int, uint32_t> mm(511 * 4, 513 * 4, 519 * 4);
    // Matmul<int, uint32_t> mm(32, 32, 24);
    // mm.compute(); // comp refC by cublas?
    // mm.print();


    for (int m = 1; m < 256; m++)
    {
        for (int k = 1; k < 188; k++)
        {
            for (int n = 1; n < 300; n++)
            {
                std::cout << "m=" << m*4 << ",k=" << k*4 << ",n=" << n*4 << std::endl;
                Matmul<int, uint32_t> mm(m*4, k*4, n*4);
                mm.compute();
            }
        }
    }
    return 0;
}