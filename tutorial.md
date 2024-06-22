


# Basics

We will show how to set things up using docker.

To get started, we first pre-install the environment using the NVIDIA Container Toolkit to avoid manual environment configuration:

```
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it \
      -v /home/msmelyan/Tutorials/Inference/:/home/msmelyan/Tutorials/Inference/ \
      -w /home/msmelyan/Tutorials/Inference/ \
      nvidia/cuda:12.3.0-devel-ubuntu22.04
       
```

Once inside the container, do

```
apt-get update
apt-get install vim
pip install -U "huggingface_hub[cli]" 
```

Here I am mounting my local directory /home/msmelyan/Tutorials/Inference/ as docker directory under the same name. When container starts, it cd to this directory.

# TensorRT

1. Once inside docker container using command in 1, install TensorRT-LLM.
```
# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check installation
python3 -c "import tensorrt_llm"
```

2. Clone TensorRT-LLM repo to get  examples/ directory which contains python scripts we will need later. **DO NOT** pip install any of the requriements.txt inside this directory as it will interfere with your current instalation of tensort_llm from above.

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```


3. Here
   
```
 
```




# Triton

# References

[Markdown Tutorial](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

[NVIDIA Blog Tutorial on TensorRT and Triton](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
