# XLA compilation of Stable Diffusion in TensorFlow

This repository provides code to serialize the different models involved in Stable Diffusion as SavedModels and to compile them with XLA. As result of the XLA-compiled concrete functions, we can obtain a good amount of speedup in the inference process.

We use the Stable Diffusion model [shipped](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/) from KerasCV.

**Table of content**

* [Results](#results)
* [Steps](#steps)
* [Running the benchmark](#running-the-benchmark)
* [Benchmark details](#details-of-the-benchmark)
* [Gotchas](#gotchas)
* [Acknowledgements](#acknowledgements)

## Results 

## Steps 

We first isolate the sub-models involved in Stable Diffusion and serialize them as
stand-alone SavedModels:

* Text encoder
* Diffusion model aka UNet
* Decoder

The SavedModel also includes their respective computations. For example SavedModel of the text encoder includes the processing of prompt context and the unconditional context. Similarly, SavedModel of the UNet includes the computations for the diffusion 
process. 

For the serialization, just run `serialize_savedmodels.py`. 

Once the SavedModels are generated, we load them as concrete functions and XLA-compile them before running inference. We include the complete code for this in `benchmark.py`. 

## Running the benchmark

## Details of the benchmark

The benchmarks were run on an `a2-highgpu-1g` [instance](https://cloud.google.com/compute/docs/gpus#a100-gpus). 

## Gotchas

## Acknowledgements