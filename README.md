# BFS ( Big F***** Server )
This is the configuration for an AI backend targetting the following hardware:

- 8 x Nvidia H100 94 GB NVL GPUs
- 2 x AMD EPYC 9755 CPUs
- About 1.5 TB DDR5 ECC Registered RAM

Currently, this cofiguration implements three LLMs running via vLLM which are accesse via a `litellm` proxy server:

- GPT-OSS-120b for deep reasoning tasks
- GPT-OSS-20b for classification tasks
- Qwen2.5-VL-32B-Instruct for processing multiple file types.

## LiteLLM
LiteLLM is used to provide a unified gateway to access these models. Istead of making requests to a particular model at `http://localhost:8000`, we instead target a "genre" of model. For example, targeting the "large model" will send requests to GPT-OSS-120b. Targetting the ""vision model" will forward requests to Qwen2.5 and similarly with "small" and gpt-oss-20b

The purpose behind this is so that the underlying model can change without causing existing AI applications to break.

## Goals

Ultimately, the goal is to make this machine the ultimate AI backend able to generate photos, videos, audio, complete transcription tasks, etc.

The larger plan is to use Nvidia's Triton Inference server for loading/unloading models across this hardware.

In lieu of vLLM, the plan is to ultimately use backends such as TensorRT, ONNX, PyTorth, OpenVINO and TensorFlow.
