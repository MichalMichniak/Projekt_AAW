#ifndef SKELETONIZATION_HPP
#define SKELETONIZATION_HPP

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"

#define INPUT_IMAGE "Skeletonization_Input.bmp"
#define OUTPUT_IMAGE "Skeletonization_Output.bmp"

#define GROUP_SIZE 256

using namespace appsdk;

class Skeletonization{
    cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
    cl_double kernelTime;               /**< time taken to run kernel and read result back */
    cl_uchar4* inputImageData;          /**< Input bitmap data to device */
    cl_uchar4* outputImageData;         /**< Output from device */

    cl::Context context;                            /**< CL context */
    std::vector<cl::Device> devices;                /**< CL device list */
    std::vector<cl::Device> device;                 /**< CL device to be used */
    std::vector<cl::Platform> platforms;            /**< list of platforms */
    cl::Image2D inputImage2D;                       /**< CL Input image2d */
    cl::Image2D outputImage2D;                      /**< CL Output image2d */
    cl::CommandQueue commandQueue;                  /**< CL command queue */
    cl::Program program;                            /**< CL program  */
    cl::Kernel kernel1;                              /**< CL kernel */
    cl::Kernel kernel2;                              /**< CL kernel */
    cl::Kernel kernel3;                              /**< CL kernel */
    cl::Kernel kernel4;                              /**< CL kernel */
    cl::Kernel kernel_check;                              /**< CL kernel */




    cl_uchar* verificationOutput;       /**< Output array for reference implementation */

    SDKBitMap inputBitmap;   /**< Bitmap class object */
    uchar4* pixelData;       /**< Pointer to image data */
    cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
    cl_uint width;                      /**< Width of image */
    cl_uint height;                     /**< Height of image */
    cl_bool byteRWSupport;
    size_t kernelWorkGroupSize1;         /**< Group Size returned by kernel */
    size_t kernelWorkGroupSize2;         /**< Group Size returned by kernel */
    size_t kernelWorkGroupSize3;         /**< Group Size returned by kernel */
    size_t kernelWorkGroupSize4;         /**< Group Size returned by kernel */
    size_t kernelWorkGroupSize_check;         /**< Group Size returned by kernel */
    size_t blockSizeX;                  /**< Work-group size in x-direction */
    size_t blockSizeY;                  /**< Work-group size in y-direction */
    int iterations;                     /**< Number of iterations for kernel execution */
    int imageSupport;

    SDKTimer    *sampleTimer;      /**< SDKTimer object */
public:

    CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

    /**
    * Read bitmap image and allocate host memory
    * @param inputImageName name of the input file
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int readInputImage(std::string inputImageName);

    /**
    * Write to an image file
    * @param outputImageName name of the output file
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int writeOutputImage(std::string outputImageName);

    /**
    * Constructor
    * Initialize member variables
    */
    Skeletonization()
        : inputImageData(NULL),
            outputImageData(NULL),
            verificationOutput(NULL),
            byteRWSupport(true)
    {
        sampleArgs = new CLCommandArgs();
        sampleTimer = new SDKTimer();
        sampleArgs->sampleVerStr = SAMPLE_VERSION;
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
        blockSizeX = GROUP_SIZE;
        blockSizeY = 1;
        iterations = 1;
        imageSupport = 0;
    }

    ~Skeletonization()
    {
    }

    /**
    * Allocate image memory and Load bitmap file
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setupSobelFilterImage();

    /**
     * Override from SDKSample, Generate binary image of given kernel
     * and exit application
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int genBinaryImage();

    /**
    * OpenCL related initialisations.
    * Set up Context, Device list, Command Queue, Memory buffers
    * Build CL kernel program executable
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setupCL();

    /**
    * Set values for kernels' arguments, enqueue calls to the kernels
    * on to the command queue, wait till end of kernel execution.
    * Get kernel start and end time if timing is enabled
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int runCLKernels();

    /**
    * Reference CPU implementation of Binomial Option
    * for performance comparison
    */
    void sobelFilterImageCPUReference();

    /**
    * Override from SDKSample. Print sample stats.
    */
    void printStats();

    /**
    * Override from SDKSample. Initialize
    * command line parser, add custom options
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int initialize();

    /**
    * Override from SDKSample, adjust width and height
    * of execution domain, perform all sample setup
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setup();

    /**
    * Override from SDKSample
    * Run OpenCL Sobel Filter
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int run();

    /**
    * Override from SDKSample
    * Cleanup memory allocations
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int cleanup();

    /**
    * Override from SDKSample
    * Verify against reference implementation
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int verifyResults();
};





#endif