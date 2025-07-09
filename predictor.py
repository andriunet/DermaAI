import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from io import BytesIO
import subprocess

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

def descargar_modelo_derm_foundation():
    derm_model_folder = "derm_foundation_model"
    if not os.path.exists(derm_model_folder):
        print("☁️ Descargando modelo desde Google Drive...")
        file_id = "10UwkOM0ON36QETxOxfSynrh9Udu5anNG"
        subprocess.run(["gdown", "--folder", f"https://drive.google.com/drive/folders/{file_id}", "--output", derm_model_folder])

def predict_skin_condition_local(image_path: str,
                                  derm_model_path: str = "derm_foundation_model",
                                  classifier_path: str = "derm_found_merge_6_class_ovr_logistic_v1.pkl"):
    descargar_modelo_derm_foundation()

    print("📥 Cargando modelo Derm Foundation...")
    derm_model = tf.saved_model.load(derm_model_path)
    derm_infer = derm_model.signatures["serving_default"]

    print("📥 Cargando clasificador...")
    xgb_model = joblib.load(classifier_path)

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
    prediction = xgb_model.predict(embedding_vector)[0]
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