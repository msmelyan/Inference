


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
pip install jupyter
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

3. Retrive the model weights

   We will play with llama-2, but other models can be downloaded similarly.

   TensorRT-LLM is a library for LLM inference, and so to use it, you must supply a set of trained weights. You can either use your own model weights trained in some framework, or pull a set of pretrained weights from repositories like the HuggingFace Hub.

   We will pull the weights and tokenizer files for the chat-tuned variant of the 7B parameter Llama 2 model from the HuggingFace Hub.

   We will work inside /home/msmelyan/Tutorials/Inference/TensorRT-LLM/examples/llama directory which already contains necassery scripts for saving model checkpoint

   ```
   cd /home/msmelyan/Tutorials/Inference/TensorRT-LLM/examples/llama/
   git lfs install
   git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   ```

   Usage of this model is subject to a particular license. To download the necessary files, agree to the terms and authenticate with Hugging Face.


5. Save the state of your running Docker container so you can reuse it with all installed packages and configurations 
   
   ```
   docker ps # to find the running container_id
   docker commit <container_id> llm_inf_tutorial_image
   ```
   Once this is done, next time you can start the container as follows:

   ```
   docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it \
      -v /home/msmelyan/Tutorials/Inference/:/home/msmelyan/Tutorials/Inference/ \
      -w /home/msmelyan/Tutorials/Inference/ \
      llm_inf_tutorial_image
   ```
   All of previously installed packages will be preserved.

6. Compiling the model

   The next step in the process is to compile the model into a TensorRT engine. For this, you need the model weights as well as a model definition written in the TensorRT-LLM Python API.

   Before you start, 

   The TensorRT-LLM repository contains a wide variety of predefined model architectures. For this post, you use the included Llama model definition instead of writing your own. This is a minimal example of some of the optimizations available in TensorRT-LLM.

   We will do our work from /home/msmelyan/Tutorials/Inference/TensorRT-LLM/examples/llama directory.

   ```
   # Log in to huggingface-cli
   # You can get your token from huggingface.co/settings/token
   huggingface-cli login --token *****

   cd /home/msmelyan/Tutorials/Inference/TensorRT-LLM/examples/llama directory:
 
   # Build the LLaMA 7B model using a single GPU and BF16.
   python3 convert_checkpoint.py --model_dir ./Llama-2-7b-chat-hf \
                              --output_dir ./tllm_checkpoint_1gpu_bf16 \
                              --dtype bfloat16
 
   trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/7B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16
   ```
   
   When you create the model definition with the TensorRT-LLM API, you build a graph of operations from NVIDIA TensorRT primitives that form the layers of your neural network. These operations map to specific GPU cuda kernels. The TensorRT compiler can sweep through the graph to choose the best kernel for each operation and available GPU. Crucially, it can also identify patterns in the graph where multiple operations are good candidates for being fused into a single kernel. This reduces the required amount of memory movement and the overhead of launching multiple GPU kernels.

   TensorRT also compiles the graph of operations into a single CUDA Graph that can be launched all at one time, further reducing the kernel launch overhead.

   The TensorRT compiler is extremely powerful for fusing layers and increasing execution speed, but there are some complex layer fusions—like FlashAttention—that involve interleaving many operations together and which can’t be automatically discovered. For those, you can explicitly replace parts of the graph with plugins at compile time.

   In this example, you include the gpt_attention plug-in, which implements a FlashAttention-like fused attention kernel, and the gemm plug-in, which performs matrix multiplication with FP32 accumulation. You also call out your desired precision for the full model as FP16, matching the default precision of the weights that you downloaded from HuggingFace.

   Here’s what this script produces when you finish running it. In the /tmp/llama/7B/trt_engines/bf16/1-gpu folder, there are now the following files:

   - Llama_float16_tp1_rank0.engine: The main output of the build script, containing the executable graph of operations with the model weights embedded.
config.json: Includes detailed information about the model, like its general structure and precision, as well as information about which plug-ins were incorporated into the engine.
   - model.cache: Caches some of the timing and optimization information from model compilation, making successive builds quicker.

8. Run the model

   So, now that you’ve got your model engine, what can you do with it?

   The engine file contains the information that you need for executing the model, but LLM usage in practice requires much more than a single forward pass through the model. TensorRT-LLM includes a highly optimized C++ runtime for executing built LLM engines and managing processes like sampling tokens from the model output, managing the KV cache, and batching requests together.

   You can use that runtime directly to execute the model locally, or you can use the TensorRT-LLM runtime backend for NVIDIA Triton Inference Server to serve the model for multiple users.

   To run the model locally, execute the following command, assuming you are still in /home/msmelyan/Tutorials/Inference/TensorRT-LLM/examples/llama directory:
   
   ```
   python3 ../run.py  \
   --engine_dir=./tmp/llama/7B/trt_engines/bf16/1-gpu \
   --max_output_len 100 \
   --tokenizer_dir meta-llama/Llama-2-7b-chat-hf \
   --input_text "How do I count to nine in French?"
   ```

   It will take time to load the TRT engine, but once it's loaded, it will start doing inference for this query, generating tokens. For longer responses, you may adjust --max_output_len length to a larger value.

# Triton

# References

[Markdown Tutorial](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

[NVIDIA Blog Tutorial on TensorRT and Triton](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)

[Turbocharging Meta Llama 3 Performance with NVIDIA TensorRT-LLM and NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/turbocharging-meta-llama-3-performance-with-nvidia-tensorrt-llm-and-nvidia-triton-inference-server/)
