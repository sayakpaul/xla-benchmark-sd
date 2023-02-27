from keras_cv.models.stable_diffusion.constants import _ALPHAS_CUMPROD
import tensorflow as tf
from .common_constants import MAX_PROMPT_LENGTH, HIDDEN_DIM, IMG_HEIGHT, IMG_WIDTH

ALPHAS_CUMPROD_tf = tf.constant(_ALPHAS_CUMPROD)

SIGNATURE_DICT = {
    "context": tf.TensorSpec(shape=[None, MAX_PROMPT_LENGTH, HIDDEN_DIM], dtype=tf.float32, name="context"),
    "unconditional_context": tf.TensorSpec(
        shape=[None, MAX_PROMPT_LENGTH, HIDDEN_DIM], dtype=tf.float32, name="unconditional_context"
    ),
    "num_steps": tf.TensorSpec(shape=[], dtype=tf.int32, name="num_steps"),
    "unconditional_guidance_scale": tf.TensorSpec(
        shape=[], dtype=tf.float32, name="unconditional_guidance_scale"
    )
}


def diffusion_model_exporter(model: tf.keras.Model):
    @tf.function
    def get_timestep_embedding(timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(
            -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    @tf.function(input_signature=[SIGNATURE_DICT])
    def serving_fn(inputs):
        img_height = tf.cast(tf.math.round(IMG_HEIGHT / 128) * 128, tf.int32)
        img_width = tf.cast(tf.math.round(IMG_WIDTH / 128) * 128, tf.int32)

        num_steps = inputs["num_steps"]
        context = inputs["context"]
        unconditional_context = inputs["unconditional_context"]
        unconditional_guidance_scale = inputs["unconditional_guidance_scale"]

        latent = tf.random.normal((NUM_IMAGES_TO_GEN, img_height // 8, img_width // 8, 4))

        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas = tf.map_fn(lambda t: ALPHAS_CUMPROD_tf[t], timesteps, dtype=tf.float32)
        alphas_prev = tf.concat([[1.0], alphas[:-1]], 0)

        index = num_steps - 1
        latent_prev = None
        for timestep in timesteps[::-1]:
            latent_prev = latent
            t_emb = get_timestep_embedding(timestep, NUM_IMAGES_TO_GEN)
            unconditional_latent = model(
                [latent, t_emb, unconditional_context], training=False
            )
            latent = model([latent, t_emb, context], training=False)
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - tf.math.sqrt(1 - a_t) * latent) / tf.math.sqrt(a_t)
            latent = (
                latent * tf.math.sqrt(1.0 - a_prev) + tf.math.sqrt(a_prev) * pred_x0
            )
            index = index - 1

        return {"latent": latent}

    return serving_fn