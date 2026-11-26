"""Demo app: drop a chest X-ray, see the predicted class and a Grad-CAM heatmap.

Usage:
    python app.py                       # opens http://127.0.0.1:7860
    python app.py --model path/to/model.keras

This is a demonstration of the trained model, NOT a medical device. The model
was trained on one public dataset whose classes come from different sources
(see docs/dataset_audit.md), so it can be wrong on X-rays from elsewhere.
"""

import argparse
from pathlib import Path

import gradio as gr
import pandas as pd

from chestxray.predict import DEFAULT_MODEL, Predictor, read_rgb

DISCLAIMER = """
**Not a medical device.** This is a student project. The model was trained on
one public Kaggle dataset in which every class comes from a different source,
so its answers say little about X-rays from other hospitals. Never use it for
a diagnosis.
"""

DESCRIPTION = """
# Chest X-ray classifier
Drop a frontal chest X-ray (JPG or PNG). The model (MobileNetV2, fine-tuned)
gives the probability of **COVID-19, Normal, Pneumonia and Tuberculosis**.
The **Grad-CAM** heatmap shows which regions pushed the prediction towards the
predicted class: red = strong influence. A trustworthy prediction should be
driven by the lungs, not by text markers or the image border.
"""


def build_app(predictor: Predictor) -> gr.Blocks:
    """Create the two-tab interface around a loaded predictor."""

    def classify(image):
        if image is None:
            return None, None
        result = predictor.predict(image)
        return result.probabilities, result.heatmap_overlay

    def classify_files(files):
        if not files:
            return pd.DataFrame()
        rows = []
        for file in files:
            path = Path(file if isinstance(file, str) else file.name)
            try:
                result = predictor.predict(read_rgb(path), explain=False)
            except OSError as error:
                rows.append({"file": path.name, "prediction": f"error: {error}"})
                continue
            rows.append(
                {
                    "file": path.name,
                    "prediction": result.predicted,
                    "confidence": round(result.probabilities[result.predicted], 3),
                    **{k: round(v, 3) for k, v in result.probabilities.items()},
                }
            )
        return pd.DataFrame(rows)

    with gr.Blocks(title="Chest X-ray classifier") as app:
        gr.Markdown(DESCRIPTION)
        gr.Markdown(DISCLAIMER)
        with gr.Tab("One image"):
            with gr.Row():
                image_in = gr.Image(type="numpy", label="Chest X-ray", height=380)
                heatmap_out = gr.Image(
                    label="Grad-CAM (224 x 224 model input)", height=380
                )
            label_out = gr.Label(
                num_top_classes=len(predictor.classes), label="Prediction"
            )
            image_in.change(classify, inputs=image_in, outputs=[label_out, heatmap_out])
        with gr.Tab("Several images"):
            files_in = gr.File(
                file_count="multiple", file_types=["image"], label="X-ray files"
            )
            table_out = gr.Dataframe(label="Predictions", wrap=True)
            files_in.change(classify_files, inputs=files_in, outputs=table_out)
    return app


def main() -> None:
    """Load the model and start the app."""
    parser = argparse.ArgumentParser(description="Chest X-ray classifier demo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(Predictor(args.model)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
