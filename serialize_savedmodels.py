import keras_cv
import tensorflow as tf

from exporters import (decoder_exporter, diffusion_model_exporter,
                       text_encoder_exporter)
from exporters.common_constants import (DECODER_PATH, DIFFUSION_MODEL_PATH,
                                        IMG_HEIGHT, IMG_WIDTH,
                                        MAX_PROMPT_LENGTH, TEXT_ENCODER_PATH)


def load_sd_models():
    """Loads the sub-models of Stable Diffusion."""
    text_encoder = keras_cv.models.stable_diffusion.text_encoder.TextEncoder(
        MAX_PROMPT_LENGTH
    )
    diffusion_model = keras_cv.models.stable_diffusion.diffusion_model.DiffusionModel(
        IMG_HEIGHT, IMG_WIDTH, MAX_PROMPT_LENGTH
    )
    decoder = keras_cv.models.stable_diffusion.decoder.Decoder(IMG_HEIGHT, IMG_WIDTH)
    return text_encoder, diffusion_model, decoder


def main():
    print("Loading the sub-models of Stable Diffusion...")
    text_encoder, diffusion_model, decoder = load_sd_models()

    print("Serializing the models as SavedModels...")
    tf.saved_model.save(
        diffusion_model,
        DIFFUSION_MODEL_PATH,
        signatures={"serving_default": diffusion_model_exporter(diffusion_model)},
    )
    print("Diffusion model saved to: {DIFFUSION_MODEL_PATH}")
    tf.saved_model.save(
        text_encoder,
        TEXT_ENCODER_PATH,
        signatures={"serving_default": text_encoder_exporter(text_encoder)},
    )
    print("Text encoder saved to: {TEXT_ENCODER_PATH}")
    tf.saved_model.save(
        decoder,
        DECODER_PATH,
        signatures={"serving_default": decoder_exporter(decoder)},
    )
    print("Decoder saved to: {DECODER_PATH}")


if __name__ == "__main__":
    main()
