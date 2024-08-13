import io
import torch
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import BlipForConditionalGeneration, AutoProcessor

app = Flask(__name__)
CORS(app)

model_path = "Trained Models/IMG-CAP-GEN-BEST.h5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading pretrained model
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
# Load the processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

if device == "cuda":
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
else:
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()


@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    if caption.endswith("."):
        caption = caption.capitalize()
    else:
        caption = caption.capitalize() + "."
    print(caption)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    # Change the host to '0.0.0.0' to make the app accessible from other devices
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(debug=True)
