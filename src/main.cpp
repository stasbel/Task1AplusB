#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    if (platforms.empty()) {
        std::cout << "Platforms not found" << std::endl;
        return 1;
    }

    cl_platform_id platform = platforms.front();
    cl_int error = 0;

    cl_uint gpuDevicesCount = 0;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuDevicesCount);
    if (error != CL_DEVICE_NOT_FOUND) OCL_SAFE_CALL(error);

    std::vector<cl_device_id> devices(gpuDevicesCount);
    if (gpuDevicesCount > 0)
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDevicesCount, devices.data(), nullptr));
    else {
        std::cout << "GPU devices not found" << std::endl;
        cl_uint cpuDevicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuDevicesCount));
        devices.resize(cpuDevicesCount);
        if (cpuDevicesCount > 0)
            OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, cpuDevicesCount, devices.data(), nullptr));
    }

    if (devices.empty()) {
        std::cout << "GPU and CPU devices not found" << std::endl;
        return 1;
    }

    cl_device_id device = devices.front();
    std::vector<cl_device_id> deviceAsVector;
    deviceAsVector.push_back(device);

    cl_context context = clCreateContext(nullptr, 1, deviceAsVector.data(), nullptr, nullptr, &error);
    OCL_SAFE_CALL(error);
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &error);
    OCL_SAFE_CALL(error);

    unsigned int n = 100 * 1000 * 1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    cl_mem aBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n, nullptr, &error);
    OCL_SAFE_CALL(error);
    error = clEnqueueWriteBuffer(commandQueue, aBuf, CL_TRUE, 0, sizeof(float) * n, as.data(), 0, nullptr, nullptr);
    OCL_SAFE_CALL(error);

    cl_mem bBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n, nullptr, &error);
    OCL_SAFE_CALL(error);
    error = clEnqueueWriteBuffer(commandQueue, bBuf, CL_TRUE, 0, sizeof(float) * n, bs.data(), 0, nullptr, nullptr);
    OCL_SAFE_CALL(error);

    cl_mem cBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &error);
    OCL_SAFE_CALL(error);
    error = clEnqueueReadBuffer(commandQueue, cBuf, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr);
    OCL_SAFE_CALL(error);
    std::cout << "Buffers successfully created" << std::endl;

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    std::vector<const char *> sourcePointers;
    sourcePointers.push_back(kernel_sources.c_str());
    std::vector<size_t> sourceLengths;
    sourceLengths.push_back(kernel_sources.length());
    cl_program program = clCreateProgramWithSource(
            context, static_cast<cl_uint>(sourcePointers.size()), sourcePointers.data(), sourceLengths.data(), &error
    );
    OCL_SAFE_CALL(error);

    error = clBuildProgram(program, static_cast<cl_uint>(devices.size()), devices.data(), nullptr, nullptr, nullptr);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
        std::vector<char> log(logSize, 0);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr));

        if (logSize > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
    }
    OCL_SAFE_CALL(error);

    std::string kernelName = "aplusb";
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &error);
    OCL_SAFE_CALL(error);

    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(aBuf), &aBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(bBuf), &bBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cBuf), &cBuf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
    }

    {
        size_t wg_size = 128, gw_size = (n + wg_size - 1) / wg_size * wg_size;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event kernel_complete;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, nullptr, &gw_size, nullptr, 0, nullptr, &kernel_complete
            ));
            OCL_SAFE_CALL(clWaitForEvents(1, &kernel_complete));
            t.nextLap();
        }

        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << (n / t.lapAvg()) / 1000000000 << std::endl;
        std::cout << "VRAM bandwidth: " << ((double) 3 * n * sizeof(float)) / (1 << 30) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(
                    commandQueue, cBuf, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr
            ));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bw: " << ((sizeof(float) * n) / t.lapAvg()) / (1 << 30) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
