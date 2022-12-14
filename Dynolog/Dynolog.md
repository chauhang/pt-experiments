# Dynolog 

Dynolog is a lightweight monitoring daemon for heterogeneous CPU-GPU systems. It supports both always-on performance monitoring, as well as deep-dive profiling modes. The latter can be activated by making a remote procedure call to the daemon.

### Install DCGMI

#### For Ubuntu LTS

Set up the CUDA network repository meta-data, GPG key

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

#### Install DCGM 

```bash
sudo apt-get update \
&& sudo apt-get install -y datacenter-gpu-manager 
```

For other OS refer: https://developer.nvidia.com/dcgm#Downloads

### Install Pytorch

#### Create a conda environment

```bash
conda create -n dynolog python=3.8
Conda activate dynolog

conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

conda install mkl mkl-include
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

#### Clone pytorch

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0

cd pytorch 
cd third_party/kineto/libkineto
git checkout main
cd -
```

Disable the USE_LITE_INTERPRETER_PROFILER setting. One way to do this is change the option in pytorch/CMakeLists.txt.

```bash
if(USE_LITE_INTERPRETER_PROFILER)
- string(APPEND CMAKE_CXX_FLAGS " -DEDGE_PROFILER_USE_KINETO")
+ # string(APPEND CMAKE_CXX_FLAGS " -DEDGE_PROFILER_USE_KINETO")
endif()
```

#### Build Pytorch

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

### Build Dynolog

#### Clone Repository

```bash
git clone https://github.com/facebookincubator/dynolog.git
git submodule update --init
```

#### Install Deps

```bash
conda install cmake ninja
conda install -c conda-forge "rust=1.64.0"

cd dynolog
./scripts/build.sh
```

### Start Dynolog

```bash
./build/dynolog/src/dynolog --enable_ipc_monitor --enable_gpu_monitor --dcgm_lib_path=/usr/lib/x86_64-linux-gnu/libdcgm.so
```

Logs:

```bash
I20221122 06:54:17.413156  2515 Main.cpp:116] Starting dynolog min, version 0.0.2
I20221122 06:54:17.413231  2515 SimpleJsonServer.cpp:82] Listening to connections on port 1778
I20221122 06:54:17.413245  2515 SimpleJsonServer.cpp:229] Launching RPC thread
I20221122 06:54:17.413319  2515 Main.cpp:129] Starting IPC Monitor
I20221122 06:54:17.413337  2517 SimpleJsonServer.cpp:207] Waiting for connection.
I20221122 06:54:17.413434  2515 LibkinetoConfigManager.cpp:298] Process count for job ID 0: 0
I20221122 06:54:17.413343  2515 IPCMonitor.cpp:29] Kineto config manager : active processes = 0
I20221122 06:54:17.413436  2518 LibkinetoConfigManager.cpp:60] Starting LibkinetoConfigManager runloop
I20221122 06:54:17.413568  2520 Main.cpp:91] Setting up DCGM (GPU)  monitoring.
I20221122 06:54:17.413623  2520 DcgmGroupInfo.cpp:105] Creating DCGM instance with fields: 100
I20221122 06:54:17.413801  2521 Main.cpp:74] Running kernel monitor loop : interval = 60 s.
I20221122 06:54:17.413938  2521 Logger.cpp:42] Logging : 1 values
I20221122 06:54:17.413954  2521 Logger.cpp:43] time = 1970-01-01T00:00:00.000Z data = {"uptime":5889}
I20221122 06:54:17.414077  2520 DcgmApiStub.cpp:146] Loaded dcgm dynamic library
I20221122 06:54:17.434123  2520 DcgmGroupInfo.cpp:152] Added group id 2
I20221122 06:54:17.434144  2520 DcgmGroupInfo.cpp:162] Found 1 supported devices, with id:
I20221122 06:54:17.434156  2520 DcgmGroupInfo.cpp:167] Successfully add device: 0
I20221122 06:54:17.434177  2520 DcgmGroupInfo.cpp:202] Added field group 4 to group 2
I20221122 06:54:17.434190  2520 DcgmGroupInfo.cpp:212] Watching DCGM fields at interval (ms) = 10000
I20221122 06:54:17.437126  2520 Main.cpp:96] Running DCGM loop : interval = 10 s.
I20221122 06:54:17.437140  2520 Main.cpp:98] DCGM fields: 100
I20221122 06:54:17.437152  2520 DcgmGroupInfo.cpp:292] fieldGroupIds_ size: 1
I20221122 06:54:17.439757  2520 DcgmGroupInfo.cpp:302] Got 1 GPU records
I20221122 06:54:17.439790  2520 DcgmGroupInfo.cpp:306] Got 1 values for entity: EntityGroupId: 1 EntityId: 0
I20221122 06:54:17.439813  2520 Logger.cpp:42] Logging : 3 values
```

### Running the pytorch program with Dynolog logging

```bash
echo "ENABLE_IPC_FABRIC=YES" > ~/libkineto.conf
KINETO_CONFIG=~/libkineto.conf KINETO_USE_DAEMON=1 python3 scripts/pytorch/linear_model_example.py
```

Logs:

```bash
I20221122 11:49:48.443688 49132 Main.cpp:74] Running kernel monitor loop : interval = 60 s.
I20221122 11:49:48.443925 49132 Logger.cpp:42] Logging : 36 values
I20221122 11:49:48.443943 49132 Logger.cpp:43] time = 2022-11-22T11:49:48.443Z data = {"cpu_i":"78.951","cpu_n_ms":0,"cpu_s":"1.138","cpu_s_ms":2720,"cpu_u":"19.911","cpu_u_ms":47610,"cpu_util":"21.049","cpu_w_ms":0,"cpu_x_ms":0,"cpu_y_ms":0,"cpu_z_ms":0,"rx_bytes_docker0":0,"rx_bytes_ens5":5641,"rx_bytes_lo":0,"rx_drops_docker0":0,"rx_drops_ens5":0,"rx_drops_lo":0,"rx_errors_docker0":0,"rx_errors_ens5":0,"rx_errors_lo":0,"rx_packets_docker0":0,"rx_packets_ens5":59,"rx_packets_lo":0,"tx_bytes_docker0":0,"tx_bytes_ens5":10309,"tx_bytes_lo":0,"tx_drops_docker0":0,"tx_drops_ens5":0,"tx_drops_lo":0,"tx_errors_docker0":0,"tx_errors_ens5":0,"tx_errors_lo":0,"tx_packets_docker0":0,"tx_packets_ens5":64,"tx_packets_lo":0,"uptime":23620}
I20221122 11:49:48.473130 49131 DcgmGroupInfo.cpp:292] fieldGroupIds_ size: 1
I20221122 11:49:48.473758 49131 DcgmGroupInfo.cpp:302] Got 1 GPU records
I20221122 11:49:48.473775 49131 DcgmGroupInfo.cpp:306] Got 3 values for entity: EntityGroupId: 1 EntityId: 0
I20221122 11:49:48.473798 49131 Logger.cpp:42] Logging : 5 values
I20221122 11:49:48.473806 49131 Logger.cpp:43] time = 1970-01-01T00:00:00.000Z data = {"dcgm_error":0,"device":0,"gpu_frequency_mhz":300,"gpu_memory_utilization":0,"gpu_power_draw":"15.797"}
I20221122 11:49:58.473210 49131 DcgmGroupInfo.cpp:292] fieldGroupIds_ size: 1
I20221122 11:49:58.473978 49131 DcgmGroupInfo.cpp:302] Got 1 GPU records
I20221122 11:49:58.474004 49131 DcgmGroupInfo.cpp:306] Got 3 values for entity: EntityGroupId: 1 EntityId: 0
I20221122 11:49:58.474037 49131 Logger.cpp:42] Logging : 5 values
I20221122 11:49:58.474052 49131 Logger.cpp:43] time = 1970-01-01T00:00:00.000Z data = {"dcgm_error":0,"device":0,"gpu_frequency_mhz":1590,"gpu_memory_utilization":0,"gpu_power_draw":"34.790"}
I20221122 11:50:08.473274 49131 DcgmGroupInfo.cpp:292] fieldGroupIds_ size: 1
I20221122 11:50:08.473944 49131 DcgmGroupInfo.cpp:302] Got 1 GPU records
I20221122 11:50:08.473963 49131 DcgmGroupInfo.cpp:306] Got 3 values for entity: EntityGroupId: 1 EntityId: 0
I20221122 11:50:08.473999 49131 Logger.cpp:42] Logging : 5 values
I20221122 11:50:08.474012 49131 Logger.cpp:43] time = 1970-01-01T00:00:00.000Z data = {"dcgm_error":0,"device":0,"gpu_frequency_mhz":1590,"gpu_memory_utilization":0,"gpu_power_draw":"32.915"}
```
