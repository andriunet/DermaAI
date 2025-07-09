import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from io import BytesIO
import gdown

# Variables globales
derm_infer = None
xgb_model = None

# Etiquetas y descripciones
CONDITION_LABELS = {
    0: 'allergic contact dermatitis',
    1: 'eczema',
    2: 'folliculitis',
    3: 'normal',
    4: 'psoriasis',
    5: 'urticaria'
}

CONDITION_DESCRIPTIONS = {
    "allergic contact dermatitis": "Skin reaction caused by contact with allergens.",
    "eczema": "Common skin condition causing itchy, inflamed patches.",
    "folliculitis": "Inflammation of hair follicles.",
    "psoriasis": "Chronic skin disease causing red, scaly patches.",
    "urticaria": "Also known as hives; itchy, raised welts on skin."
}

def ensure_model_files():
    model_folder = "derm_foundation_model"

    folder_id = os.environ.get("DERM_MODEL_DRIVE_ID")
    if not os.path.exists(model_folder):
        if folder_id:
            print("☁️ Descargando modelo Derm Foundation...")
            gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", quiet=False)
        else:
            raise RuntimeError("❌ ERROR: DERM_MODEL_DRIVE_ID no está definido")

def cargar_modelos():
    global derm_infer, xgb_model

    if derm_infer is None or xgb_model is None:
        ensure_model_files()

        print("📥 Cargando modelo Derm Foundation...")
        derm_model = tf.saved_model.load("derm_foundation_model")
        derm_infer = derm_model.signatures["serving_default"]

        print("📥 Cargando clasificador .pkl local...")
        xgb_model = joblib.load("derm_found_modelo_v1.pkl")

        print("✅ Modelos cargados en memoria.")

def predict_skin_condition_local(image_path: str):
    cargar_modelos()

    print(f"🖼️ Cargando imagen: {image_path}")
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buf = BytesIO()
    image.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))}
    )).SerializeToString()

    print("🧠 Generando embedding...")
    output = derm_infer(inputs=tf.constant([example]))
    embedding_vector = output['embedding'].numpy().flatten().reshape(1, -1)

    print("📊 Realizando predicción...")
    probabilities = xgb_model.predict_proba(embedding_vector)[0]

    probs = np.array(probabilities)
    top2_idx = probs.argsort()[-2:][::-1]
    top2_probs = probs[top2_idx]

    results = []
    for i in top2_idx:
        condition = CONDITION_LABELS.get(i, f"Unknown-{i}")
        prob_percent = round((probs[i] * 100) / top2_probs.sum(), 1)
        description = CONDITION_DESCRIPTIONS.get(condition, "")
        results.append((condition.title(), f"{prob_percent}%", description))

    return results
