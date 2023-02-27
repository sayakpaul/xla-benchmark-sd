from keras_cv.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
import tensorflow as tf
from .common_constants import MAX_PROMPT_LENGTH, NUM_IMAGES_TO_GEN

POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
UNCONDITIONAL_TOKENS = tf.convert_to_tensor([_UNCONDITIONAL_TOKENS], dtype=tf.int32)

SIGNATURE_DICT = {
    "tokens": tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name="tokens"),
}

def text_encoder_exporter(model: tf.keras.Model):
    @tf.function(input_signature=[SIGNATURE_DICT])
    def serving_fn(inputs):
        # context
        encoded_text = model([inputs["tokens"], POS_IDS], training=False)
        encoded_text = tf.squeeze(encoded_text)

        if tf.rank(encoded_text) == 2:
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), NUM_IMAGES_TO_GEN, axis=0
            )

        # unconditional context
        unconditional_context = model([UNCONDITIONAL_TOKENS, POS_IDS], training=False)

        unconditional_context = tf.repeat(unconditional_context, NUM_IMAGES_TO_GEN, axis=0)
        return {"context": encoded_text, "unconditional_context": unconditional_context}

    return serving_fn