import io
import glob
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

st.set_page_config(page_title="MNIST Autoencoder Denoising", page_icon="🧹", layout="wide")


# ---- Custom layer from the notebook (needed only if the best model is the Contractive AE) ----
@keras.utils.register_keras_serializable(package="lab04")
class ContractiveDense(layers.Layer):
    def __init__(self, units, lam=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units, self.lam = units, lam

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=(input_shape[-1], self.units),
                                      initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")

    def call(self, x):
        h = tf.sigmoid(tf.matmul(x, self.kernel) + self.bias)
        dh = h * (1.0 - h)
        w_sq = tf.reduce_sum(tf.square(self.kernel), axis=0)
        penalty = tf.reduce_mean(tf.reduce_sum(tf.square(dh) * w_sq, axis=1))
        self.add_loss(self.lam * penalty)
        return h

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "lam": self.lam})
        return config


@st.cache_resource
def load_model():
    paths = glob.glob("models/*.keras")
    if not paths:
        st.error("No .keras model found in the models/ folder.")
        st.stop()
    path = paths[0]
    model = keras.models.load_model(path, compile=False)
    name = path.split("/")[-1].replace(".keras", "").replace("_", " ")
    kind = "img" if len(model.input_shape) == 4 else "flat"   # same logic as fmt() in the notebook
    return model, kind, name


def fmt(x, kind):
    return x.reshape(len(x), 784) if kind == "flat" else x.reshape(len(x), 28, 28, 1)


def add_noise(x, nf, seed):
    r = np.random.default_rng(seed)
    return np.clip(x + nf * r.normal(0.0, 1.0, x.shape), 0.0, 1.0).astype("float32")


def to_png_bytes(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return buf.getvalue()


def upscale(arr_uint8, size=280):
    # nearest-neighbour keeps the 28x28 pixels crisp (same as image-rendering: pixelated)
    return Image.fromarray(arr_uint8).resize((size, size), Image.NEAREST)


model, kind, model_name = load_model()

st.title(f"MNIST Autoencoder Denoising — {model_name}")
st.write("Upload a handwritten digit image (28×28 grayscale works best). "
         "The model deployed here was the best performer on the lab's PSNR results.")

with st.sidebar:
    st.header("Settings")
    is_noisy = st.checkbox("This image is already noisy", value=False,
                           help="Leave unchecked to add synthetic noise for testing.")
    nf = st.slider("Synthetic noise factor (used only if unchecked above)",
                   0.1, 0.5, 0.3, 0.1, disabled=is_noisy)

uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L").resize((28, 28))
    x = np.asarray(pil_img).astype("float32") / 255.0
    if x.mean() > 0.5:          # white background -> invert, as in the notebook
        x = 1.0 - x

    if not is_noisy:
        if "seed" not in st.session_state:
            st.session_state.seed = np.random.randint(0, 1_000_000)
        if st.button("🔄 Resample noise"):
            st.session_state.seed = np.random.randint(0, 1_000_000)
        x = add_noise(x[None, ...], nf, seed=st.session_state.seed)[0]

    pred = model.predict(fmt(x[None, ...], kind), verbose=0)[0]
    noisy_disp = (np.clip(x, 0, 1) * 255).astype("uint8")
    clean_disp = (np.clip(pred.reshape(28, 28), 0, 1) * 255).astype("uint8")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input (as processed)")
        st.image(upscale(noisy_disp))
    with c2:
        st.subheader("Denoised output")
        st.image(upscale(clean_disp))
        st.download_button("⬇️ Download denoised image", to_png_bytes(clean_disp),
                           file_name="denoised.png", mime="image/png")
else:
    st.info("Upload an image to get started.")
