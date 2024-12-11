#include "utils.cuh"
#define VNAME(name)(#name)

void printMatrix(float Mat[][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << Mat[i][j] << ",";
        }
        std::cout << "\n" << std::endl;
    }
}

CudaStamp* set_clock_start() {

    CudaStamp* new_stamp = new struct CudaStamp;
    //cudaEvent_t stop, start;

    cudaEventCreate(&new_stamp->start);       //Create Event

    cudaEventCreate(&new_stamp->stop);

    cudaEventRecord(new_stamp->start, 0);    //record current time

    //new_stamp->start = &start;
    //new_stamp->stop = &stop;

    return new_stamp;
}

float set_clock_end(CudaStamp* stamp, float time_elapsed) {
 
    cudaEventRecord(stamp->stop, 0);    //record current time

    //cudaEventSynchronize(*stamp->start);    //Waits for an event to complete.

    cudaEventSynchronize(stamp->stop);    //Waits for an event to complete. 

    cudaEventElapsedTime(&time_elapsed, stamp->start, stamp->stop);    //calculate time diff

    cudaEventDestroy(stamp->start);    //destory the event

    cudaEventDestroy(stamp->stop);

    return time_elapsed;
}

void get_device_info() {
    int support, device = 0;
    cudaDeviceProp my_prop;

    cudaError_t err = cudaDeviceGetAttribute(&support, cudaDevAttrComputePreemptionSupported, device);
    assert(err == cudaSuccess);
    err = cudaGetDeviceProperties(&my_prop, device);
    assert(err == cudaSuccess);
    if (support) {
        cout << VNAME(my_prop.name) << ": " << my_prop.name << endl;
        //cout << VNAME(my_prop.uuid) << ": " << my_prop.uuid << endl;
        cout << VNAME(my_prop.luid) << ": " << my_prop.luid << endl;
        cout << VNAME(my_prop.luidDeviceNodeMask) << ": " << my_prop.luidDeviceNodeMask << endl;
        cout << VNAME(my_prop.totalGlobalMem) << ": " << my_prop.totalGlobalMem << endl;
        cout << VNAME(my_prop.sharedMemPerBlock) << ": " << my_prop.sharedMemPerBlock << endl;
        cout << VNAME(my_prop.regsPerBlock) << ": " << my_prop.regsPerBlock << endl;
        cout << VNAME(my_prop.warpSize) << ": " << my_prop.warpSize << endl;
        cout << VNAME(my_prop.memPitch) << ": " << my_prop.memPitch << endl;
        cout << VNAME(my_prop.maxThreadsPerBlock) << ": " << my_prop.maxThreadsPerBlock << endl;
        cout << VNAME(my_prop.maxThreadsDim) << ": " << my_prop.maxThreadsDim << endl;
        cout << VNAME(my_prop.maxGridSize) << ": " << my_prop.maxGridSize << endl;
        cout << VNAME(my_prop.clockRate) << ": " << my_prop.clockRate << endl;
        cout << VNAME(my_prop.totalConstMem) << ": " << my_prop.totalConstMem << endl;
        cout << VNAME(my_prop.major) << ": " << my_prop.major << endl;
        cout << VNAME(my_prop.minor) << ": " << my_prop.minor << endl;
        cout << VNAME(my_prop.textureAlignment) << ": " << my_prop.textureAlignment << endl;
        cout << VNAME(my_prop.texturePitchAlignment) << ": " << my_prop.texturePitchAlignment << endl;
        cout << VNAME(my_prop.deviceOverlap) << ": " << my_prop.deviceOverlap << endl;
        cout << VNAME(my_prop.multiProcessorCount) << ": " << my_prop.multiProcessorCount << endl;
        cout << VNAME(my_prop.kernelExecTimeoutEnabled) << ": " << my_prop.kernelExecTimeoutEnabled << endl;
        cout << VNAME(my_prop.integrated) << ": " << my_prop.integrated << endl;
        cout << VNAME(my_prop.canMapHostMemory) << ": " << my_prop.canMapHostMemory << endl;
        cout << VNAME(my_prop.computeMode) << ": " << my_prop.computeMode << endl;
        cout << VNAME(my_prop.maxTexture1D) << ": " << my_prop.maxTexture1D << endl;
        cout << VNAME(my_prop.maxTexture1DMipmap) << ": " << my_prop.maxTexture1DMipmap << endl;
        cout << VNAME(my_prop.maxTexture1DLinear) << ": " << my_prop.maxTexture1DLinear << endl;
        cout << VNAME(my_prop.maxTexture2D[2]) << ": " << my_prop.maxTexture2D[2] << endl;
        cout << VNAME(my_prop.maxTexture2DMipmap[2]) << ": " << my_prop.maxTexture2DMipmap[2] << endl;
        cout << VNAME(my_prop.maxTexture2DLinear[3]) << ": " << my_prop.maxTexture2DLinear[3] << endl;
        cout << VNAME(my_prop.maxTexture2DGather[2]) << ": " << my_prop.maxTexture2DGather[2] << endl;
        cout << VNAME(my_prop.maxTexture3D[3]) << ": " << my_prop.maxTexture3D[3] << endl;
        cout << VNAME(my_prop.maxTexture3DAlt[3]) << ": " << my_prop.maxTexture3DAlt[3] << endl;
        cout << VNAME(my_prop.maxTextureCubemap) << ": " << my_prop.maxTextureCubemap << endl;
        cout << VNAME(my_prop.maxTexture1DLayered[2]) << ": " << my_prop.maxTexture1DLayered[2] << endl;
        cout << VNAME(my_prop.maxTexture2DLayered[3]) << ": " << my_prop.maxTexture2DLayered[3] << endl;
        cout << VNAME(my_prop.maxTextureCubemapLayered[2]) << ": " << my_prop.maxTextureCubemapLayered[2] << endl;
        cout << VNAME(my_prop.maxSurface1D) << ": " << my_prop.maxSurface1D << endl;
        cout << VNAME(my_prop.maxSurface2D[2]) << ": " << my_prop.maxSurface2D[2] << endl;
        cout << VNAME(my_prop.maxSurface3D[3]) << ": " << my_prop.maxSurface3D[3] << endl;
        cout << VNAME(my_prop.maxSurface1DLayered[2]) << ": " << my_prop.maxSurface1DLayered[2] << endl;
        cout << VNAME(my_prop.maxSurface2DLayered[3]) << ": " << my_prop.maxSurface2DLayered[3] << endl;
        cout << VNAME(my_prop.maxSurfaceCubemap) << ": " << my_prop.maxSurfaceCubemap << endl;
        cout << VNAME(my_prop.maxSurfaceCubemapLayered[2]) << ": " << my_prop.maxSurfaceCubemapLayered[2] << endl;
        cout << VNAME(my_prop.surfaceAlignment) << ": " << my_prop.surfaceAlignment << endl;
        cout << VNAME(my_prop.concurrentKernels) << ": " << my_prop.concurrentKernels << endl;
        cout << VNAME(my_prop.ECCEnabled) << ": " << my_prop.ECCEnabled << endl;
        cout << VNAME(my_prop.pciBusID) << ": " << my_prop.pciBusID << endl;
        cout << VNAME(my_prop.pciDeviceID) << ": " << my_prop.pciDeviceID << endl;
        cout << VNAME(my_prop.pciDomainID) << ": " << my_prop.pciDomainID << endl;
        cout << VNAME(my_prop.tccDriver) << ": " << my_prop.tccDriver << endl;
        cout << VNAME(my_prop.asyncEngineCount) << ": " << my_prop.asyncEngineCount << endl;
        cout << VNAME(my_prop.unifiedAddressing) << ": " << my_prop.unifiedAddressing << endl;
        cout << VNAME(my_prop.memoryClockRate) << ": " << my_prop.memoryClockRate << endl;
        cout << VNAME(my_prop.memoryBusWidth) << ": " << my_prop.memoryBusWidth << endl;
        cout << VNAME(my_prop.l2CacheSize) << ": " << my_prop.l2CacheSize << endl;
        cout << VNAME(my_prop.persistingL2CacheMaxSize) << ": " << my_prop.persistingL2CacheMaxSize << endl;
        cout << VNAME(my_prop.maxThreadsPerMultiProcessor) << ": " << my_prop.maxThreadsPerMultiProcessor << endl;
        cout << VNAME(my_prop.streamPrioritiesSupported) << ": " << my_prop.streamPrioritiesSupported << endl;
        cout << VNAME(my_prop.globalL1CacheSupported) << ": " << my_prop.globalL1CacheSupported << endl;
        cout << VNAME(my_prop.localL1CacheSupported) << ": " << my_prop.localL1CacheSupported << endl;
        cout << VNAME(my_prop.sharedMemPerMultiprocessor) << ": " << my_prop.sharedMemPerMultiprocessor << endl;
        cout << VNAME(my_prop.regsPerMultiprocessor) << ": " << my_prop.regsPerMultiprocessor << endl;
        cout << VNAME(my_prop.managedMemory) << ": " << my_prop.managedMemory << endl;
        cout << VNAME(my_prop.isMultiGpuBoard) << ": " << my_prop.isMultiGpuBoard << endl;
        cout << VNAME(my_prop.multiGpuBoardGroupID) << ": " << my_prop.multiGpuBoardGroupID << endl;
        cout << VNAME(my_prop.hostNativeAtomicSupported) << ": " << my_prop.hostNativeAtomicSupported << endl;
        cout << VNAME(my_prop.singleToDoublePrecisionPerfRatio) << ": " << my_prop.singleToDoublePrecisionPerfRatio << endl;
        cout << VNAME(my_prop.pageableMemoryAccess) << ": " << my_prop.pageableMemoryAccess << endl;
        cout << VNAME(my_prop.concurrentManagedAccess) << ": " << my_prop.concurrentManagedAccess << endl;
        cout << VNAME(my_prop.computePreemptionSupported) << ": " << my_prop.computePreemptionSupported << endl;
        cout << VNAME(my_prop.canUseHostPointerForRegisteredMem) << ": " << my_prop.canUseHostPointerForRegisteredMem << endl;
        cout << VNAME(my_prop.cooperativeLaunch) << ": " << my_prop.cooperativeLaunch << endl;
        cout << VNAME(my_prop.cooperativeMultiDeviceLaunch) << ": " << my_prop.cooperativeMultiDeviceLaunch << endl;
        cout << VNAME(my_prop.sharedMemPerBlockOptin) << ": " << my_prop.sharedMemPerBlockOptin << endl;
        cout << VNAME(my_prop.pageableMemoryAccessUsesHostPageTables) << ": " << my_prop.pageableMemoryAccessUsesHostPageTables << endl;
        cout << VNAME(my_prop.directManagedMemAccessFromHost) << ": " << my_prop.directManagedMemAccessFromHost << endl;
        cout << VNAME(my_prop.maxBlocksPerMultiProcessor) << ": " << my_prop.maxBlocksPerMultiProcessor << endl;
        cout << VNAME(my_prop.accessPolicyMaxWindowSize) << ": " << my_prop.accessPolicyMaxWindowSize << endl;
        cout << VNAME(my_prop.reservedSharedMemPerBlock) << ": " << my_prop.reservedSharedMemPerBlock << endl;
    }

    return;
}

bool is_array2D_equil(float A[][N], float B[][N]) {
    assert(sizeof(A[0]) == sizeof(B[0]));
    bool is_equal = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (abs(A[i][j] - B[i][j]) >= 1e-3) {
                is_equal = false;
                // cout << "i: " << i << ",j:" << j << ",A:" << A[i][j] << ",B:"<< B[i][j] << endl;
            }
        }
    }
    return is_equal;
}