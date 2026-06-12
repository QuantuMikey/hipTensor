/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "hip_device.hpp"
#include <hiptensor/internal/hiptensor_utility.hpp>

namespace hiptensor
{
    HipDevice::HipDevice()
        : mDeviceId(-1)
        , mGcnArch(hipGcnArch_t::UNSUPPORTED_ARCH)
        , mWarpSize(hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
        , mSharedMemSize(0)
        , mCuCount(0)
        , mMaxFreqMhz(0)
    {
        // CSC patch (Piece 0, CPU-node operability): the upstream constructor
        // wrapped these two queries in CHECK_HIP_ERROR, which terminates the
        // host process (exit(EXIT_FAILURE)) when no ROCm-capable device is
        // present. That turned a recoverable "no GPU here" condition into a
        // fatal abort for any caller that touches hipTensor on a CPU-only
        // node. Instead, leave the sentinel state (mDeviceId = -1,
        // UNSUPPORTED_ARCH) on failure; downstream API entry points already
        // compare device identity / architecture and surface clean
        // HIPTENSOR_STATUS errors for an invalid device.
        if(hipGetDevice(&mDeviceId) != hipSuccess)
        {
            (void)hipGetLastError();
            mDeviceId = -1;
            return;
        }
        if(hipGetDeviceProperties(&mProps, mDeviceId) != hipSuccess)
        {
            (void)hipGetLastError();
            mDeviceId = -1;
            return;
        }

        mArch = mProps.arch;

        std::string deviceName(mProps.gcnArchName);

        if(deviceName.find("gfx908") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX908;
        }
        else if(deviceName.find("gfx90a") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX90A;
        }
        else if(deviceName.find("gfx940") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX940;
        }
        else if(deviceName.find("gfx941") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX941;
        }
        else if(deviceName.find("gfx942") != std::string::npos)
        {
            mGcnArch = hipGcnArch_t::GFX942;
        }

        switch(mProps.warpSize)
        {
        case hipWarpSize_t::Wave64:
            mWarpSize = mProps.warpSize;
        default:;
        }

        mSharedMemSize = mProps.sharedMemPerBlock;
        mCuCount       = mProps.multiProcessorCount;
        mMaxFreqMhz    = static_cast<int>(static_cast<double>(mProps.clockRate) / 1000.0);
    }

    hipDevice_t HipDevice::getDeviceId() const
    {
        return mDeviceId;
    }

    hipDeviceProp_t HipDevice::getDeviceProps() const
    {
        return mProps;
    }

    hipDeviceArch_t HipDevice::getDeviceArch() const
    {
        return mArch;
    }

    HipDevice::hipGcnArch_t HipDevice::getGcnArch() const
    {
        return mGcnArch;
    }

    int HipDevice::warpSize() const
    {
        return mWarpSize;
    }

    int HipDevice::sharedMemSize() const
    {
        return mSharedMemSize;
    }

    int HipDevice::cuCount() const
    {
        return mCuCount;
    }

    int HipDevice::maxFreqMhz() const
    {
        return mMaxFreqMhz;
    }

    bool HipDevice::supportsF64() const
    {
        return (mGcnArch == HipDevice::hipGcnArch_t::GFX90A
                || mGcnArch == HipDevice::hipGcnArch_t::GFX940
                || mGcnArch == HipDevice::hipGcnArch_t::GFX941
                || mGcnArch == HipDevice::hipGcnArch_t::GFX942);
    }

    // Need to check the host device target support statically before hip modules attempt
    // to load any kernels. Not safe to proceed if the host device is unsupported.
    struct HipStaticDeviceGuard
    {
        static bool testSupportedDevice()
        {
            auto device = HipDevice();

            // CSC patch (Piece 0, CPU-node operability): when NO HIP device is
            // present at all, the patched HipDevice constructor leaves the
            // sentinel mDeviceId == -1 instead of exit()ing. This static guard
            // runs at shared-library load (static initialization), i.e. at
            // `import qiskit_aer`; calling exit() here kills any host process
            // that merely links hipTensor on a CPU-only node. Distinguish the
            // two cases: no device -> load is fine, GPU kernels are simply
            // never launched (callers gate on device availability and
            // hiptensorCreate returns a clean error status); a REAL device
            // with an unsupported arch -> keep the upstream hard stop, since
            // HIP modules would otherwise attempt to load kernels for it.
            if(device.getDeviceId() < 0)
            {
                std::cerr << "hipTensor: no ROCm-capable device detected at load; "
                             "GPU contraction paths are disabled."
                          << std::endl;
                return false;
            }

            if((device.getGcnArch() == HipDevice::hipGcnArch_t::UNSUPPORTED_ARCH)
               || (device.warpSize() == HipDevice::hipWarpSize_t::UNSUPPORTED_WARP_SIZE))
            {
                std::cerr << "Cannot proceed: unsupported host device detected. Exiting."
                          << std::endl;
                exit(EXIT_FAILURE);
            }
            return true;
        }
        static bool sResult;
    };

    bool HipStaticDeviceGuard::sResult = HipStaticDeviceGuard::testSupportedDevice();

} // namespace hiptensor
