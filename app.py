from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_KEY = '8b4dRagvAbkG7YmsojDMOSFP1W7IuHRz8bwLplv5'
TEMPERATURE = 0.8
NUM_CLASSES = 101
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classes
class_data = torch.load('models/food101_classes.pth', map_location=device)
class_names = class_data['classes']

# Load models
inc_model = models.inception_v3(weights=None, aux_logits=True)
inc_model.fc = nn.Linear(inc_model.fc.in_features, NUM_CLASSES)
inc_model.load_state_dict(torch.load('models/InceptionV3_finetuned.pth', map_location=device))
inc_model.to(device).eval()

res_model = models.resnet50(weights=None)
res_model.fc = nn.Linear(res_model.fc.in_features, NUM_CLASSES)
res_model.load_state_dict(torch.load('models/ResNet50_finetuned.pth', map_location=device))
res_model.to(device).eval()

# Transforms
inc_tf = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
res_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def temperature_scaled_prediction(model, img_tensor, temp=1.0):
    with torch.no_grad():
        out = model(img_tensor)
        logits = out if not isinstance(out, tuple) else out[0]
        logits = logits / temp
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
        return class_names[idx.item()], conf.item() * 100

def clean_food_name_for_usda(food_name):
    return food_name.replace("_", " ").lower().strip()

def fetch_usda_calories(food_name):
    cleaned = clean_food_name_for_usda(food_name)
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "api_key": API_KEY,
        "query": cleaned,
        "pageSize": 3
    }
    results = []
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        for food in data.get("foods", []):
            desc = food.get("description", "N/A")
            cal = next((n.get("value") for n in food.get("foodNutrients", [])
                        if n.get("nutrientName", "").lower() == "energy"), None)
            size = food.get("servingSize")
            unit = food.get("servingSizeUnit", "")
            serving = f"{size} {unit}".strip() if size else "N/A"
            results.append({
                "description": desc.title(),
                "calories": f"{cal} kcal" if cal is not None else "N/A",
                "serving": serving
            })
    except Exception as e:
        print("❌ USDA API error:", str(e))
        results.append({"description": "API Error", "calories": "N/A", "serving": "N/A"})
    return results[:3]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file: return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath).convert('RGB')
        inc_img = inc_tf(image).unsqueeze(0).to(device)
        res_img = res_tf(image).unsqueeze(0).to(device)

        inc_class, inc_conf = temperature_scaled_prediction(inc_model, inc_img, TEMPERATURE)
        res_class, res_conf = temperature_scaled_prediction(res_model, res_img, TEMPERATURE)

        if inc_conf > res_conf:
            final_class, final_conf, model_used = inc_class, inc_conf, "Inception-v3"
        else:
            final_class, final_conf, model_used = res_class, res_conf, "ResNet-50"

        usda_data = fetch_usda_calories(final_class)

        # Save everything in session
        session['uploaded_image'] = filename
        session['prediction'] = {
            "inc": [inc_class, inc_conf],
            "res": [res_class, res_conf],
            "final": [final_class, final_conf, model_used]
        }
        session['usda'] = usda_data

        return redirect(url_for('index'))

    # GET method: Load data from session
    uploaded_image = session.pop('uploaded_image', None)
    prediction = session.pop('prediction', None)
    usda = session.pop('usda', None)

    return render_template('index.html',
                           uploaded_image=uploaded_image,
                           prediction=prediction,
                           usda=usda)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
