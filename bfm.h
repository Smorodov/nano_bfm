#include "cnpy.h"
#include <iostream>
#include <queue> // очередь
#include <stack> // стек
#include <vector>
#include <set>
#include <omp.h>
#include "cuda_runtime_api.h"
#include <cublasLt.h>


#define COUT_VAR(x) std::cout << #x"=" << x << std::endl;
#define SHOW_IMG(x) cv::namedWindow(#x);cv::imshow(#x,x);cv::waitKey(20);
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
class bfm
{
    //---------------------------------------------------------------------------
    // Simple class for triangle mesh normals computarion
    //---------------------------------------------------------------------------
    class NormalsComputer
    {
    private:
        // vector of vertices shared triangles
        // vertex normals compured by averaging face normals
        std::vector < std::set<unsigned int> > sharedTriangles;
        std::vector<unsigned int> faces;
    public:
        //-------------------
        // constructor
        //-------------------
        NormalsComputer(std::vector<unsigned int>& faces)
        {
            this->faces.assign(faces.begin(), faces.end());
            // need to compute once per mesh
            computeSharedTriangles(faces, sharedTriangles);
        }
        //-------------------
        // destructor
        //-------------------
        ~NormalsComputer()
        {

        }
        //--------------------------------------------------------------------------
        // compute vertex normals
        // input: vertices as array of floats in v1x,v1y,v1z,...vmx,vmy,vmz format
        //--------------------------------------------------------------------------
        void getVertexNormals(unsigned int nVertices, float* vertices, float* verticesNormals)
        {
            computeVertexNormals(faces.size(), nVertices, faces.data(), vertices, verticesNormals);
        }
        //--------------------------------------------------------------------------
        // same for std::vector type 
        //--------------------------------------------------------------------------
        void getVertexNormals(std::vector<float>& vertices, std::vector<float>& verticesNormals)
        {
            computeVertexNormals(faces, vertices, verticesNormals);
        }
        //--------------------------------------------------------------------------
        // same for face normals
        //--------------------------------------------------------------------------
        void getFaceNormals(std::vector<float>& vertices, std::vector<float>& normals)
        {
            computeFaceNormals(faces, vertices, normals);
        }
    private:
        //--------------------------------------------------------------------------
        // compute trianges shared by each vertex
        //--------------------------------------------------------------------------
        void computeSharedTriangles(std::vector<unsigned int>& faces, std::vector < std::set<unsigned int> >& result)
        {
            unsigned int max_ind = *std::max_element(faces.begin(), faces.end()) + 1;
            result.resize(max_ind);

            for (unsigned int i = 0; i < faces.size() / 3; ++i)
            {
                unsigned int a = faces.data()[i * 3 + 0];
                unsigned int b = faces.data()[i * 3 + 1];
                unsigned int c = faces.data()[i * 3 + 2];

                if (a == b || a == c || b == c)
                {
                    std::cout << "Incorrect triangle !" << std::endl;
                }
                result[a].insert(i);
                result[b].insert(i);
                result[c].insert(i);
            }
        }
        //--------------------------------------------------------------------------
        // compute one face normsl
        //--------------------------------------------------------------------------
        void computeNormal(float* a, float* b, float* c, float* normal)
        {
            float v1[3], v2[3];
            v1[0] = b[0] - a[0];
            v1[1] = b[1] - a[1];
            v1[2] = b[2] - a[2];

            v2[0] = b[0] - c[0];
            v2[1] = b[1] - c[1];
            v2[2] = b[2] - c[2];

            normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
            normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
            normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
            float nx = normal[0];// / norm;
            float ny = normal[1];// / norm;
            float nz = normal[2];// / norm;
            float norm = sqrt(nx * nx + ny * ny + nz * nz);
            //
            normal[0] = -nx;
            normal[1] = -ny;
            normal[2] = -nz;
        }
        //--------------------------------------------------------------------------
        // compute all face normals
        //--------------------------------------------------------------------------
        void computeFaceNormals(std::vector<unsigned int>& faces, std::vector<float>& vertices, std::vector<float>& normals)
        {
            normals.clear();
            normals.resize(faces.size(), 0);
#pragma omp parallel for
            for (unsigned int i = 0; i < faces.size() / 3; ++i)
            {
                float* a;
                float* b;
                float* c;
                float* N;
                unsigned int a_ind = faces[3 * i + 0];
                unsigned int b_ind = faces[3 * i + 1];
                unsigned int c_ind = faces[3 * i + 2];
                if (a_ind == b_ind || a_ind == c_ind || b_ind == c_ind)
                {
                    std::cout << "Incorrect triangle !" << std::endl;
                }
                a = vertices.data() + a_ind * 3;
                b = vertices.data() + b_ind * 3;
                c = vertices.data() + c_ind * 3;
                computeNormal(a, b, c, normals.data() + i * 3);
            }
        }
        //--------------------------------------------------------------------------
        // compute vertex normals by averaging shared face normals
        //--------------------------------------------------------------------------
        void computeVertexNormals(std::vector<unsigned int>& faces, std::vector<float>& vertices, std::vector<float>& result)
        {
            result.resize(vertices.size());
            std::fill(result.begin(), result.end(), 0);
            std::vector<float> faceNormals;
            computeFaceNormals(faces, vertices, faceNormals);
            for (unsigned int i = 0; i < result.size()/3; ++i)
            {
                std::set<unsigned int>::iterator it;
                for (it = sharedTriangles[i].begin(); it != sharedTriangles[i].end(); ++it)
                {
                    unsigned int t = *it*3;
                    unsigned int sz = faceNormals.size();
                    result[i * 3 + 0] += faceNormals[t + 0];
                    result[i * 3 + 1] += faceNormals[t + 1];
                    result[i * 3 + 2] += faceNormals[t + 2];
                }
                float n = sqrt(result[i * 3 + 0] * result[i * 3 + 0] + result[i * 3 + 1] * result[i * 3 + 1] + result[i * 3 + 2] * result[i * 3 + 2]);
                if (n > 0)
                {
                    result[i * 3 + 0] /= n;
                    result[i * 3 + 1] /= n;
                    result[i * 3 + 2] /= n;
                }
                else
                {
                    std::cout << "zero normal" << std::endl;
                }
            }
        }
        //--------------------------------------------------------------------------
        // smae as before for pointers
        //--------------------------------------------------------------------------
        void computeVertexNormals(unsigned int nFaces, unsigned int nVertices, unsigned int* faces, float* vertices, float* result)
        {
            std::vector<unsigned int> vfaces(nFaces);
            std::vector<float> vvertices(nVertices);
            memcpy(vfaces.data(), faces, nFaces * sizeof(unsigned int));
            memcpy(vvertices.data(), vertices, nVertices * sizeof(float));
            std::vector<float> vresult;
            computeVertexNormals(vfaces, vvertices, vresult);
            memcpy(result, vresult.data(), vresult.size() * sizeof(float));
        }
    };    

public:
    // -----------------------------
    // cublas related
    // -----------------------------
    cublasLtHandle_t ltHandle;
    cudaStream_t stream;
    size_t workspaceSize;
    void* workspace;

    float alpha, beta;
    
    void initCublas(void);
    void copyDataToDevice();
    void copyDataFromDevice();
    void streamSynchronize();
    void deinitCublas();
    // -----------------------------   
    std::vector<float> AhostShape, BhostShape;
    std::vector<float> ChostShape, biasHostShape;    
    float* AdevShape, * BdevShape;
    float* CdevShape, * biasDevShape;
    // -----------------------------   
    std::vector<float> AhostTex, BhostTex;
    std::vector<float> ChostTex, biasHostTex;
    float* AdevTex, * BdevTex;
    float* CdevTex, * biasDevTex;
    // -----------------------------   

    int face_index_num;
    int vertex_num;
    int shape_basis_dim;
    int tex_basis_dim;
    cnpy::npz_t npz_map;
    cnpy::NpyArray tl;
    cnpy::NpyArray shapeMU; // (160470, 1)
    cnpy::NpyArray shapePC; // (160470, 199)
    cnpy::NpyArray shapeEV; // (199, 1)
    cnpy::NpyArray shapeCoeffs;
    cnpy::NpyArray Shape;
    cnpy::NpyArray vetexNormals;

    cnpy::NpyArray texMU; // (160470, 1)
    cnpy::NpyArray texPC; // (160470, 199)
    cnpy::NpyArray texEV; // (199, 1)    
    cnpy::NpyArray texCoeffs;
    cnpy::NpyArray Tex;
    
    std::shared_ptr<NormalsComputer> normalsComputer;
    bfm();
    ~bfm();
public:
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void loadModel(std::string filename);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void updateMesh(std::vector<float>& shape, std::vector<float>& tex);
};
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
