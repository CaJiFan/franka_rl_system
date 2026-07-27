# Installation Notes — Wipe Vision Modules

System-level dependencies that **must be installed before** running
`pip install -r requirements.txt`.

---

## 1 · Orbbec SDK (`vision_module.py`)

`pyorbbecsdk` is a Python binding that wraps Orbbec's native C++ SDK.
The native runtime must be present on the system first.

### Linux (Ubuntu 20.04 / 22.04)

```bash
# 1. Download the Orbbec SDK for Linux from:
#    https://github.com/orbbec/OrbbecSDK/releases
#    Pick the archive matching your Ubuntu version, e.g.:
#    OrbbecSDK_v1.x.x_linux_x86_64.tar.gz

tar -xzf OrbbecSDK_v*.tar.gz
cd OrbbecSDK_*/

# 2. Install udev rules so the camera is accessible without root
sudo cp misc/udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 3. Install the shared libraries
sudo cp lib/linux_x64/*.so* /usr/local/lib/
sudo ldconfig

# 4. Now install the Python binding
pip install pyorbbecsdk
```

> **Tip:** If `pyorbbecsdk` is not yet on PyPI for your platform, build it
> from source: <https://github.com/orbbec/pyorbbecsdk>

---

## 2 · Intel RealSense SDK (`vision_module_realsense.py`)

### Option A — pip only (usually works on Ubuntu 20.04+)

```bash
pip install pyrealsense2
```

If the pip wheel bundles the native libs you are done.
Verify with `python -c "import pyrealsense2; print(pyrealsense2.__version__)"`.

### Option B — librealsense2 system install (if Option A fails)

```bash
# Add Intel's apt repository
sudo apt-get install -y software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com \
     --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCD
sudo add-apt-repository \
     "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# Install SDK and Python bindings
sudo apt-get install -y librealsense2-dkms librealsense2-utils \
                        librealsense2-dev librealsense2-dbg

# Then install the pip wrapper
pip install pyrealsense2
```

Full docs: <https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md>

---

## 3 · ZED SDK (`vision_module_zed.py`)

The ZED SDK and its Python API (`pyzed`) must be installed at the OS level first.

### Linux (Ubuntu 20.04 / 22.04)

```bash
# 1. Download the ZED SDK installer from:
#    https://www.stereolabs.com/developers/release/
#    Choose the .run file that matches your Ubuntu and CUDA version.

chmod +x ZED_SDK_Ubuntu*_v*.run
./ZED_SDK_Ubuntu*_v*.run

# 2. Run the get_python_api.py script to install pyzed
#    (this is usually located in /usr/local/zed/ or provided in the SDK directory)
cd /usr/local/zed/
python get_python_api.py

# 3. Verify the installation
python -c "import pyzed.sl as sl; print('ZED SDK version:', sl.Camera().get_sdk_version())"
```

> **Note:** Do NOT install `pyzed` via `pip install pyzed` from PyPI, as that might pull an incompatible or unofficial package. Always use the Stereolabs provided installation script.

---

## Quick-start summary

```bash
# -- System deps (run once per machine) --

# Orbbec
sudo cp OrbbecSDK_*/lib/linux_x64/*.so* /usr/local/lib/ && sudo ldconfig
sudo cp OrbbecSDK_*/misc/udev/*.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules

# RealSense (if pip wheel is not self-contained)
sudo apt-get install -y librealsense2-dkms librealsense2-dev

# ZED
./ZED_SDK_*.run -- silent
python /usr/local/zed/get_python_api.py

# -- Python packages --
pip install -r requirements.txt
```

---

## Environment tested on

| Component | Version |
|---|---|
| Python | 3.10 |
| Ubuntu | 22.04 LTS |
| opencv-python | 4.10.x |
| numpy | 1.26.x |
| pyorbbecsdk | 1.x |
| pyrealsense2 | 2.54.x |
| pyzed | 4.x |
