import argparse
import timeit

import keras_cv
import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from tensorflow.python.saved_model import tag_constants

from exporters.common_constants import (DECODER_PATH, DIFFUSION_MODEL_PATH,
                                        HIDDEN_DIM, IMG_HEIGHT, IMG_WIDTH,
                                        LATENTS_RES, MAX_PROMPT_LENGTH,
                                        NUM_IMAGES_TO_GEN, PADDING_TOKEN,
                                        TEXT_ENCODER_PATH)

GPUS = tf.config.list_logical_devices("GPU")
TOKENIZER = SimpleTokenizer()


def load_keras_cv_sd(jit_compile=True):
    model = keras_cv.models.StableDiffusion(
        img_width=IMG_HEIGHT, img_height=IMG_WIDTH, jit_compile=jit_compile
    )
    return model


def create_concrete_fn(model_path):
    model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    return model_loaded.signatures["serving_default"]


def load_concrete_fns_sd(jit_compile=True):
    """Loads the SavedModels of SD as concrete functions."""
    df_model_fn = create_concrete_fn(DIFFUSION_MODEL_PATH)
    text_encoder_fn = create_concrete_fn(TEXT_ENCODER_PATH)
    decoder_fn = create_concrete_fn(DECODER_PATH)

    if jit_compile:
        df_model_fn = tf.function(df_model_fn, jit_compile=jit_compile)
        decoder_fn = tf.function(decoder_fn, jit_compile=jit_compile)
        # Cannot XLA-compile the text encoder. See: https://github.com/tensorflow/tensorflow/issues/59818

    # Variable bindings.
    run_inference_concrete_fn(df_model_fn, text_encoder_fn, decoder_fn)

    return df_model_fn, text_encoder_fn, decoder_fn


def run_inference_concrete_fn(df_model_fn, text_encoder_fn, decoder_fn):
    """Runs dummy inference with the concrete functions for warm-ups."""
    # Diffusion model concrete func.
    batch_size = tf.constant(NUM_IMAGES_TO_GEN)
    context = tf.random.normal((batch_size, MAX_PROMPT_LENGTH, HIDDEN_DIM))
    num_steps = tf.constant(10)
    unconditional_guidance_scale = tf.constant(10.5)

    _ = df_model_fn(
        context=context,
        num_steps=num_steps,
        unconditional_context=context,
        unconditional_guidance_scale=unconditional_guidance_scale,
    )

    # Text-encoder concrete func.
    _ = text_encoder_fn(tokens=tf.ones((batch_size, MAX_PROMPT_LENGTH), tf.int32))

    # Decoder concrete func.
    _ = decoder_fn(latent=tf.random.normal((batch_size, LATENTS_RES, LATENTS_RES, 4)))


def run_inference_kerascv_sd(model):
    """Runs dummy inference with the KerasCV model class for warm-ups."""
    _ = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def run_dummy_inference(model):
    # Concrete functions.
    if isinstance(model, tuple):
        df_model_fn, text_encoder_fn, decoder_fn = model[0], model[1], model[2]
        run_inference_concrete_fn(df_model_fn, text_encoder_fn, decoder_fn)
    else:
        run_inference_kerascv_sd(model)


def generate_images_from_text(
    model, text, unconditional_guidance_scale=7.5, num_steps=50
) -> np.ndarray:
    """Generates images from an input text prompt with the concrete functions."""
    df_model_fn, text_encoder_fn, decoder_fn = model

    num_steps = tf.constant(num_steps)
    unconditional_guidance_scale = tf.constant(unconditional_guidance_scale)

    tokens = TOKENIZER.encode(text)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    tokens = tf.convert_to_tensor([tokens], dtype=tf.int32)

    with tf.device(GPUS[0].name):
        encoded_text = text_encoder_fn(tokens=tokens)

        latents = df_model_fn(
            context=encoded_text["context"],
            num_steps=num_steps,
            unconditional_context=encoded_text["unconditional_context"],
            unconditional_guidance_scale=unconditional_guidance_scale,
        )

        decoded_images = decoder_fn(latent=latents["latent"])
    return decoded_images["generated_images"].numpy()


def main(args):
    print("Loading model / concrete funcs...")
    if not args.kerascv:
        df_model_fn, text_encoder_fn, decoder_fn = load_concrete_fns_sd(
            args.jit_compile
        )
        model = (df_model_fn, text_encoder_fn, decoder_fn)
    else:
        model = load_keras_cv_sd(args.jit_compile)

    print("Warming up...")
    run_dummy_inference(model)

    print("Benchmarking...")
    if not args.kerascv:
        runtimes = timeit.repeat(
            lambda: generate_images_from_text(model, args.prompt), number=1, repeat=25
        )
    else:
        runtimes = timeit.repeat(
            lambda: model.text_to_image(args.prompt, batch_size=NUM_IMAGES_TO_GEN),
            number=1,
            repeat=25,
        )

    print(f"Using KerasCV default model: {args.kerascv}, Using XLA: {args.jit_compile}")
    print(f"Average latency (seconds): {np.mean(runtimes)}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute otter in a rainbow whirlpool holding shells, watercolor",
    )
    parser.add_argument("--jit_compile", action="store_true")
    parser.add_argument("--kerascv", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
