import io
import os
import glob
import zipfile
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="MNIST Autoencoder Denoising", page_icon="🧹", layout="wide")

# ---------------- Green dashboard styling ----------------
st.markdown(
    """
    <style>
    h1, h2, h3, h4 { color: #064E3B !important; font-weight: 700; }
    /* upload drop-zone: light green with dashed border */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ECFAF1;
        border: 1.5px dashed #0B8A4B;
        border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #0B8A4B; color: #FFFFFF; border: none; font-weight: 600;
    }
    [data-testid="stFileUploaderDropzone"] button:hover { background-color: #087A41; }
    /* info / alert boxes in green */
    [data-testid="stAlert"] {
        background-color: #E3F6EA; color: #064E3B; border-radius: 10px;
    }
    /* buttons */
    .stButton > button, .stDownloadButton > button {
        border: 1px solid #0B8A4B; color: #0B8A4B; border-radius: 8px; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #0B8A4B; color: #FFFFFF; border-color: #0B8A4B;
    }
    /* keep pixel digits crisp */
    [data-testid="stImage"] img { image-rendering: pixelated; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    size_kb = os.path.getsize(path) / 1024
    if not zipfile.is_zipfile(path):
        st.error(f"{path} is not a valid .keras file (size: {size_kb:.1f} KB). "
                 "Re-export it from Colab and re-upload it.")
        st.stop()
    model = keras.models.load_model(path, compile=False)
    name = model.name.replace("_", " ")                       # e.g. "Denoising AE"
    kind = "img" if len(model.input_shape) == 4 else "flat"   # same logic as fmt() in the notebook
    return model, kind, name


def fmt(x, kind):
    return x.reshape(len(x), 784) if kind == "flat" else x.reshape(len(x), 28, 28, 1)


def add_noise(x, nf, seed):
    r = np.random.default_rng(seed)
    return np.clip(x + nf * r.normal(0.0, 1.0, x.shape), 0.0, 1.0).astype("float32")


def preprocess(pil_img):
    """MNIST-style preprocessing: invert, stretch contrast, crop, fit to 20x20, centre in 28x28."""
    g = np.asarray(pil_img.convert("L")).astype("float32") / 255.0
    if g.mean() > 0.5:                                  # dark ink on light paper -> invert
        g = 1.0 - g
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)      # stretch contrast
    g = np.clip((g - 0.3) / 0.7, 0, 1)                  # suppress grey background

    ys, xs = np.where(g > 0.2)
    if len(ys) == 0:                                    # nothing found -> plain resize
        return np.asarray(pil_img.convert("L").resize((28, 28))).astype("float32") / 255.0
    crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    s = 20.0 / max(h, w)                                # longest side -> 20 px, keep aspect ratio
    small = Image.fromarray((crop * 255).astype("uint8")).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    small = np.asarray(small).astype("float32") / 255.0

    canvas = np.zeros((28, 28), "float32")
    r0, c0 = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[r0:r0 + small.shape[0], c0:c0 + small.shape[1]] = small
    return canvas


def to_png_bytes(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return buf.getvalue()


def upscale(arr_uint8, size=280):
    return Image.fromarray(arr_uint8).resize((size, size), Image.NEAREST)


model, kind, model_name = load_model()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    is_noisy = st.checkbox("This image is already noisy", value=False,
                           help="Leave unchecked to add synthetic noise for testing.")
    nf = st.slider("Synthetic noise factor (used only if unchecked above)",
                   0.1, 0.5, 0.3, 0.1, disabled=is_noisy)

# ---------------- Main page ----------------
st.title(f"MNIST Autoencoder Denoising — {model_name}")
st.write("Upload a handwritten digit image (28×28 grayscale works best). "
         "The model deployed here was the best performer on the lab's PSNR results.")

uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    x = preprocess(Image.open(uploaded))

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
