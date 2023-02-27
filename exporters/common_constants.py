MAX_PROMPT_LENGTH = 77
IMG_HEIGHT = 512
IMG_WIDTH = 512
HIDDEN_DIM = 768

NUM_IMAGES_TO_GEN = 3  # Needs to be fixed otherwise the `tf.repeat()` will be complain.

LATENTS_RES = 64
PADDING_TOKEN = 49407

DIFFUSION_MODEL_PATH = "./diffusion_model/1/"
TEXT_ENCODER_PATH = "./text_encoder/1/"
DECODER_PATH = "./decoder/1/"
