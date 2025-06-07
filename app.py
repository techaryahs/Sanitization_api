from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import torch
import timm
from torch import nn
from torchvision import transforms
import cloudinary
import cloudinary.uploader
from datetime import datetime

app = Flask(__name__)

# Cloudinary configuration (🔵 replace with your own credentials)
cloudinary.config(
    cloud_name="ddk6up2ps",
    api_key="582328933293284",
    api_secret="bPiSSxfcoNA-_Cyc8ehRAfDgeDY"
)

# Load model once
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 2)

state_dict = torch.load("vit_sanitization.pth", map_location=torch.device("cpu"))
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

class_names = ["Bad", "Good"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    image = Image.open(BytesIO(file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    # Draw border and label
    border_color = (255, 0, 0) if prediction == "Bad" else (0, 255, 0)  # Red for Bad, Green for Good
    border_width = 10

    # Create a new image with border
    bordered_image = Image.new('RGB', 
                               (image.width + 2*border_width, image.height + 2*border_width), 
                               border_color)
    bordered_image.paste(image, (border_width, border_width))

    # Draw label text
    draw = ImageDraw.Draw(bordered_image)

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    text = prediction
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = (10, 10)

    draw.rectangle(
        [text_position, (text_position[0] + text_width + 10, text_position[1] + text_height + 5)],
        fill=border_color
    )
    draw.text((text_position[0] + 5, text_position[1] + 2), text, fill="white", font=font)

    # Save image temporarily in memory
    buffer = BytesIO()
    bordered_image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Upload to Cloudinary
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{now}_{prediction}.jpg"

    upload_result = cloudinary.uploader.upload(
        buffer,
        folder="sanitization_results",
        public_id=filename.split('.')[0],  # Without .jpg
        overwrite=True,
        resource_type="image"
    )

    # Get the secure URL
    image_url = upload_result.get("secure_url")

    return jsonify({
        "prediction": prediction,
        "image_url": image_url
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
