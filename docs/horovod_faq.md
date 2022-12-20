# Horovod installation issue

> ```
> Extension horovod.torch has not been built: /home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-38-x86_64-linux-gnu.so not found
> If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
>Warning! MPI libs are missing, but python applications are still avaiable.
> ```

One might fix this issue by 
```python
$ pip uninstall -y horovod
$ conda install gcc_linux-64 gxx_linux-64
$ [flags] pip install --no-cache-dir horovod
```

from [here](https://github.com/horovod/horovod/issues/656), [here](https://github.com/tlkh/ai-lab/issues/27) and [here](https://horovod.readthedocs.io/en/stable/install_include.html)

- install horovod from scratch with torch 

```python
conda create -n hd python=3.8 scipy numpy pandas -y
conda activate hd
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-11.3 /usr/local/cuda
conda install gxx_linux-64 -y
conda install cxx-compiler=1.0 -y
export TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0"
echo $TORCH_CUDA_ARCH_LIST
sudo apt-get purge -y cmake
wget -q https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
tar -zxvf cmake-3.20.2.tar.gz
cd cmake-3.20.2
./bootstrap -- -DCMAKE_USE_OPENSSL=OFF
make -j10
sudo make install
cmake --version
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export HOROVOD_NCCL_HOME=/usr/local/cuda/
export HOROVOD_NCCL_INCLUDE=/usr/local/cuda/include
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST//";8.0"/}
export HOROVOD_BUILD_CUDA_CC_LIST=${TORCH_CUDA_ARCH_LIST//";"/","}
export HOROVOD_BUILD_CUDA_CC_LIST=${HOROVOD_BUILD_CUDA_CC_LIST//"."/""}
export PATH=/usr/local/cuda/bin/:$PATH
export HOROVOD_NCCL_LIB=/usr/local/cuda/lib/
HOROVOD_NCCL_HOME=/usr/local/cuda HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1  HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_GLOO=1 pip install --no-cache-dir horovod
```

[reference 1](https://stackoverflow.com/questions/54948216/usr-lib-x86-64-linux-gnu-libstdc-so-6-version-glibcxx-3-4-21-not-found-req) and [reference 2](https://github.com/horovod/horovod/issues/401) and [reference 3](https://github.com/Lightning-AI/lightning/issues/4472) and [reference 4](https://github.com/horovod/horovod/issues/2276) and [reference 5](https://github.com/Lightning-AI/lightning/blob/master/dockers/base-cuda/Dockerfile#L105-L121) and [reference 6](https://horovod.readthedocs.io/en/stable/gpus_include.html) and [reference 7](https://horovod.readthedocs.io/en/stable/conda_include.html) and [reference 8](https://github.com/horovod/horovod/issues/3545) and [reference 9](https://github.com/KAUST-CTL/horovod-gpu-data-science-project) and [reference 10](https://kose-y.github.io/blog/2017/12/installing-cuda-aware-mpi/)