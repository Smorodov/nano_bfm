// http://rodolphe-vaillant.fr/?e=29

#include "bfm.h"
#include <iomanip>
#include <chrono>
#include <xmmintrin.h>
// http://web.archive.org/web/20150531202539/http://www.codeproject.com/Articles/4522/Introduction-to-SSE-Programming
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE2
#include <omp.h>

// --------------------------------------
//
// --------------------------------------
inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}
// --------------------------------------
//
// --------------------------------------
inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}
// --------------------------------------
//
// --------------------------------------
void print3dArray(cnpy::NpyArray& arr)
{
    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    for (size_t i = 0; i < arr.shape[0]; i++)
    {
        for (size_t j = 0; j < arr.shape[1]; j++)
        {
            for (size_t k = 0; k < arr.shape[2]; k++)
            {
                std::cout << std::setw(8) << arr.data<float>()[i * arr.shape[1] * arr.shape[2] + j * arr.shape[2] + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << " -------- " << std::endl;
    }
    std::cout << " ================= " << std::endl;
}
// --------------------------------------
//
// --------------------------------------
void print2dArray(cnpy::NpyArray& arr)
{
    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    for (size_t i = 0; i < arr.shape[0]; i++)
    {
        for (size_t j = 0; j < arr.shape[1]; j++)
        {
            std::cout << std::setw(8) << arr.data<float>()[i * arr.shape[1] + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " ================= " << std::endl;
}
//--------------------------------------------------------------------------
// Clone cnpy::NpyArray
//--------------------------------------------------------------------------
void cloneArray(cnpy::NpyArray& src, cnpy::NpyArray& dst)
{
    dst = cnpy::NpyArray(src.shape, src.word_size, src.fortran_order);
    memcpy(dst.data<unsigned char>(), src.data<unsigned char>(), src.num_bytes());
}

//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::initCublas(void)
{
    checkCublasStatus(cublasLtCreate(&ltHandle));
    //workspaceSize = 4 * (vertex_num * shape_basis_dim);
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AdevShape), vertex_num * shape_basis_dim * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BdevShape), shape_basis_dim * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CdevShape), vertex_num * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDevShape), vertex_num * sizeof(float)));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
    
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AdevTex), vertex_num * tex_basis_dim * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BdevTex), tex_basis_dim * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CdevTex), vertex_num * sizeof(float)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDevTex), vertex_num * sizeof(float)));
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));

    
    
    checkCudaStatus(cudaStreamCreate(&stream));

}
void bfm::deinitCublas()
{
    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(AdevShape));
    checkCudaStatus(cudaFree(BdevShape));
    checkCudaStatus(cudaFree(CdevShape));
    checkCudaStatus(cudaFree(biasDevShape));

    checkCudaStatus(cudaFree(AdevTex));
    checkCudaStatus(cudaFree(BdevTex));
    checkCudaStatus(cudaFree(CdevTex));
    checkCudaStatus(cudaFree(biasDevTex));


    checkCudaStatus(cudaFree(workspace));
    checkCudaStatus(cudaStreamDestroy(stream));
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::copyDataToDevice()
{
    checkCudaStatus(cudaMemcpyAsync(AdevShape, AhostShape.data(), AhostShape.size() * sizeof(AhostShape[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(BdevShape, BhostShape.data(), BhostShape.size() * sizeof(BhostShape[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(CdevShape, ChostShape.data(), ChostShape.size() * sizeof(ChostShape[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(biasDevShape, biasHostShape.data(), biasHostShape.size() * sizeof(biasHostShape[0]), cudaMemcpyHostToDevice));

    checkCudaStatus(cudaMemcpyAsync(AdevTex, AhostTex.data(), AhostTex.size() * sizeof(AhostTex[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(BdevTex, BhostTex.data(), BhostTex.size() * sizeof(BhostTex[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(CdevTex, ChostTex.data(), ChostTex.size() * sizeof(ChostTex[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(biasDevTex, biasHostTex.data(), biasHostTex.size() * sizeof(biasHostTex[0]), cudaMemcpyHostToDevice));


}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::copyDataFromDevice()
{
    checkCudaStatus(cudaMemcpyAsync(biasHostShape.data(), CdevShape, biasHostShape.size() * sizeof(biasHostShape[0]), cudaMemcpyDeviceToHost, stream));
    checkCudaStatus(cudaMemcpyAsync(biasHostTex.data(), CdevTex, biasHostTex.size() * sizeof(biasHostTex[0]), cudaMemcpyDeviceToHost, stream));
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::streamSynchronize()
{
    checkCudaStatus(cudaStreamSynchronize(stream));
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void LtSgemm(cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha, /* host pointer */
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta, /* host pointer */
    float* C,
    int ldc,
    void* workspace,
    size_t workspaceSize)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc,  CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
    
    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle,
        operationDesc,
        alpha,
        A,
        Adesc,
        B,
        Bdesc,
        beta,
        C,
        Cdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::loadModel(std::string filename)
{
    std::cout << "---------------------------" << std::endl;
    std::cout << "Loding SMPL model file : " << filename << std::endl;
    std::cout << "---------------------------" << std::endl;
    npz_map = cnpy::npz_load(filename);
    // face indices
    tl = npz_map["tl"]; // n_faces*3
    face_index_num = tl.shape[0];
    
    for (int i = 0; i < face_index_num*3; ++i)
    {
        tl.data<unsigned int>()[i]--; // rebase indices to 0
    }
    
    COUT_VAR(face_index_num);
    std::cout << "faceIndices shape ";
    printDims(tl);
    std::cout << std::endl;
    // mean mesh vertices
    shapeMU = npz_map["shapeMU"]; // n_vert*3
    vertex_num = shapeMU.shape[0]; // n_vert*3
    COUT_VAR(vertex_num);
    std::cout << "shapeMU shape:";
    printDims(shapeMU);
    std::cout << std::endl;
    // shape basis
    shapePC = npz_map["shapePC"]; // n_vert*3,shape_basis_dim
    shape_basis_dim = shapePC.shape[1];
    COUT_VAR(shape_basis_dim);
    std::cout << "shapePC shape:";
    printDims(shapePC);
    std::cout << std::endl;
    
    // shape eigenvalues
    shapeEV = npz_map["shapeEV"]; // n_vert*3,shape_basis_dim
    std::cout << "shapeEV shape:";
    printDims(shapeEV);
    std::cout << std::endl;

    // mean tex
    texMU = npz_map["texMU"]; // n_vert*3
    std::cout << "mean tex shape:";
    printDims(texMU);
    std::cout << std::endl;

    // tex principal comps
    texPC = npz_map["texPC"]; // n_vert*3,tex_basis_dim
    tex_basis_dim = texPC.shape[1];
    COUT_VAR(tex_basis_dim);
    std::cout << "texPC shape:";
    printDims(texPC);
    std::cout << std::endl;

    // tex eigenvalues
    texEV = npz_map["texEV"]; // n_vert*3,shape_basis_dim
    std::cout << "texEV shape:";
    printDims(texEV);
    std::cout << std::endl;

    // tex basis
    texPC = npz_map["texPC"];// n_vert*3,tex_basis_dim
    tex_basis_dim = texPC.shape[1];
    COUT_VAR(tex_basis_dim);
    std::cout << "texPC shape:";
    printDims(texPC);
    std::cout << std::endl;
    vetexNormals = cnpy::NpyArray(shapeMU.shape, shapeMU.word_size,shapeMU.fortran_order);
    normalsComputer = std::make_shared<NormalsComputer>(tl.as_vec<unsigned int>());
    std::cout << "---------------------------" << std::endl;
    std::cout << " Successfully loaded " << std::endl;
    std::cout << "---------------------------" << std::endl;   
    
    // 
    std::vector<float> shape(shape_basis_dim, 0);
    std::vector<float> tex(tex_basis_dim, 0);    
    cloneArray(texMU, Tex);
    cloneArray(shapeMU, Shape); 
 

    AhostShape.resize(vertex_num * shape_basis_dim);
    BhostShape.resize(shape_basis_dim);
    ChostShape.resize(vertex_num);
    biasHostShape.resize(vertex_num);
    memcpy(AhostShape.data(), shapePC.data<float>(), vertex_num * shape_basis_dim * sizeof(float));
    memcpy(BhostShape.data(), shape.data(), shape_basis_dim * sizeof(float));
    memcpy(ChostShape.data(), shapeMU.data<float>(), vertex_num * sizeof(float));
    memcpy(biasHostShape.data(), shapeMU.data<float>(), vertex_num * sizeof(float));

    AhostTex.resize(vertex_num * tex_basis_dim);
    BhostTex.resize(tex_basis_dim);
    ChostTex.resize(vertex_num);
    biasHostTex.resize(vertex_num);
    memcpy(AhostTex.data(), texPC.data<float>(), vertex_num * tex_basis_dim * sizeof(float));
    memcpy(BhostTex.data(), tex.data(), tex_basis_dim * sizeof(float));
    memcpy(ChostTex.data(), texMU.data<float>(), vertex_num * sizeof(float));
    memcpy(biasHostTex.data(), texMU.data<float>(), vertex_num * sizeof(float));


    initCublas();
    copyDataToDevice();    
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void bfm::updateMesh(std::vector<float>& shape, std::vector<float>& tex)
{ 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cloneArray(texMU, Tex);
    cloneArray(shapeMU, Shape);

    std::vector<float> scaled_shape(shape_basis_dim,0);
    for (int i = 0; i < shape.size(); ++i)
    {
        scaled_shape[i] =  shapeEV.data<float>()[i] * shape[i];
    }
    memcpy(BhostShape.data(), scaled_shape.data(), shape_basis_dim * sizeof(float));    
    
    checkCudaStatus(cudaMemcpyAsync(BdevShape, BhostShape.data(), BhostShape.size() * sizeof(BhostShape[0]), cudaMemcpyHostToDevice, stream));   
    checkCudaStatus(cudaMemcpyAsync(CdevShape, ChostShape.data(), ChostShape.size() * sizeof(ChostShape[0]), cudaMemcpyHostToDevice, stream));
    
    float alpha = 1.0f;
    float beta = 1.0f;
    
    int m = 1;
    int n = vertex_num;
    int k = shape_basis_dim;
    int lda = 1;
    int ldb = shape_basis_dim;
    int ldc = 1;

    LtSgemm(ltHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,        
        &alpha,
        BdevShape,
        lda,
        AdevShape,
        ldb,
        &beta,
        CdevShape,
        ldc,
        workspace,
        workspaceSize);


    std::vector<float> scaled_tex(tex_basis_dim, 0);
    for (int i = 0; i < tex.size(); ++i)
    {
        scaled_tex[i] = texEV.data<float>()[i] * tex[i];
    }
    memcpy(BhostTex.data(), scaled_tex.data(), tex_basis_dim * sizeof(float));

    checkCudaStatus(cudaMemcpyAsync(BdevTex, BhostTex.data(), BhostTex.size() * sizeof(BhostTex[0]), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(CdevTex, ChostTex.data(), ChostTex.size() * sizeof(ChostTex[0]), cudaMemcpyHostToDevice, stream));

    m = 1;
    n = vertex_num;
    k = tex_basis_dim;
    lda = 1;
    ldb = tex_basis_dim;
    ldc = 1;

    LtSgemm(ltHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        BdevTex,
        lda,
        AdevTex,
        ldb,
        &beta,
        CdevTex,
        ldc,
        workspace,
        workspaceSize);




    copyDataFromDevice();
    memcpy(Shape.data<float>(), biasHostShape.data(), vertex_num * sizeof(float));
    memcpy(Tex.data<float>(), biasHostTex.data(), vertex_num * sizeof(float));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Elapseed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
bfm::bfm()
{    
    
}
bfm::~bfm()
{
    deinitCublas();
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
