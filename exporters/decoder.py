import tensorflow as tf 
from ..common_constants import LATENTS_RES


SIGNATURE_DICT = {
    "latent": tf.TensorSpec(shape=[None, LATENTS_RES, LATENTS_RES, 4], dtype=tf.float32, name="latent"),
}

def decoder_exporter(model: tf.keras.Model):
    @tf.function(input_signature=[SIGNATURE_DICT])
    def serving_fn(inputs):
        latent = inputs["latent"]
        decoded = model(latent, training=False)
        decoded = ((decoded + 1) / 2) * 255
        images = tf.clip_by_value(decoded, 0, 255)
        images = tf.cast(images, tf.uint8)
        return {"generated_images": images}

    return serving_fn