import os
import io
import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

# --- Sentinel Hub Imports ---
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

# --- 1. App Initialization ---
app = FastAPI(
    title="Unified Agriculture API",
    description="Serves AI models, provides static MATLAB analysis, and offers live satellite data analysis.",
    version="2.0.0"
)

# --- CORS (useful for any frontend, including Streamlit / web) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for demo; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Configuration & Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
except NameError:
    project_root = os.path.abspath(".")

# Load .env from project root if present
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# Model configurations
MODELS_CONFIG = {
    "rice": {
        "path": os.path.join(project_root, "python", "models", "rice_leaf_model_v2.h5"),
        "classes": ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    },
    "tomato": {
        "path": os.path.join(project_root, "python", "models", "tomato_model_v1.h5"),
        "classes": [
            'Tomato_Bacterial_spot',
            'Tomato_Early_blight',
            'Tomato_Late_blight',
            'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato_healthy'
        ]
    },
    "potato": {
        "path": os.path.join(project_root, "python", "models", "potato_model_v1.h5"),
        "classes": [
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy'
        ]
    },
    "wheat": {
        "path": os.path.join(project_root, "python", "models", "wheat_model_final_version.h5"),
        "classes": ['black_rust', 'brown_rust', 'healthy', 'septoria']
    }
}

# MATLAB file paths
MATLAB_MAT_FILE = os.path.join(
    project_root, "matlab", "Vegetation_Indices", "vegetation_indices.mat"
)
MATLAB_MAPS_FOLDER = os.path.join(project_root, "matlab", "Matlab_Output")

# --- Sentinel Hub Configuration ---
config = SHConfig()

# Prefer environment variables if provided (.env or system env)
env_client_id = os.getenv("SH_CLIENT_ID")
env_client_secret = os.getenv("SH_CLIENT_SECRET")

if env_client_id:
    config.sh_client_id = env_client_id
if env_client_secret:
    config.sh_client_secret = env_client_secret

# Be explicit about token URL (default is usually this)
if not config.sh_token_url:
    config.sh_token_url = "https://services.sentinel-hub.com/oauth/token"

if not config.sh_client_id or not config.sh_client_secret:
    print("⚠️ WARNING: Sentinel Hub credentials are not configured. "
          "The /analyze-live-data endpoint will fail.")
else:
    print("✅ Sentinel Hub config loaded. Client ID starts with:",
          str(config.sh_client_id)[:15] + "...")

# --- Solution Knowledge Base ---
SOLUTION_KNOWLEDGE_BASE = {
    "Bacterial leaf blight": {
        "treatment": "Apply copper-based bactericides.",
        "prevention": "Use resistant varieties."
    },
    "Brown spot": {
        "treatment": "Apply a fungicide containing Propiconazole.",
        "prevention": "Ensure proper water drainage."
    },
    "Leaf smut": {
        "treatment": "Seed treatment with fungicides like Carboxin.",
        "prevention": "Use certified clean seeds."
    },
    "Tomato Bacterial spot": {
        "treatment": "Apply copper-based bactericides mixed with mancozeb.",
        "prevention": "Avoid overhead watering."
    },
    "Tomato Early blight": {
        "treatment": "Apply fungicides containing Chlorothalonil or Mancozeb.",
        "prevention": "Mulch around plants."
    },
    "Tomato Late blight": {
        "treatment": "Apply fungicides like Chlorothalonil or copper-based compounds.",
        "prevention": "Ensure wide spacing between plants."
    },
    "Tomato Tomato YellowLeaf Curl Virus": {
        "treatment": "No cure. Control whitefly populations.",
        "prevention": "Remove infected plants & control vectors."
    },
    "Potato Early blight": {
        "treatment": "Spray fungicides like Mancozeb or Chlorothalonil.",
        "prevention": "Practice crop rotation."
    },
    "Potato Late blight": {
        "treatment": "Apply fungicides like Mancozeb or Metalaxyl.",
        "prevention": "Use certified seed potatoes."
    },
    "black rust": {
        "treatment": "Apply fungicides like Tebuconazole.",
        "prevention": "Plant resistant varieties."
    },
    "brown rust": {
        "treatment": "Foliar fungicides are effective.",
        "prevention": "Use resistant varieties."
    },
    "septoria": {
        "treatment": "Apply fungicides. Timing is critical.",
        "prevention": "Practice crop rotation."
    },
    "healthy": {
        "treatment": "No treatment necessary.",
        "prevention": "Continue good practices."
    }
}

# --- 3. Pre-load Models and Data ---
loaded_models = {}
for crop, details in MODELS_CONFIG.items():
    if os.path.exists(details["path"]):
        loaded_models[crop] = tf.keras.models.load_model(details["path"])

if os.path.exists(MATLAB_MAT_FILE):
    mat_data = loadmat(MATLAB_MAT_FILE)
else:
    mat_data = None

print("✅ Server startup complete.")

# --- 4. Helper Functions ---
def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


def generate_health_report(index_name, data_map):
    mean_val = np.mean(data_map)
    min_val = np.min(data_map)
    max_val = np.max(data_map)
    std_dev = np.std(data_map)

    total_pixels = data_map.size
    stressed_pixels = np.sum(data_map < 0.3)
    moderate_pixels = np.sum((data_map >= 0.3) & (data_map < 0.6))
    healthy_pixels = np.sum(data_map >= 0.6)

    rows, cols = data_map.shape
    row_step, col_step = rows // 4, cols // 4
    zone_alerts, zone_means = [], []

    for i in range(4):
        zone_row = []
        for j in range(4):
            zone = data_map[i * row_step:(i + 1) * row_step,
                            j * col_step:(j + 1) * col_step]
            zone_mean = np.mean(zone)
            zone_row.append(f"{zone_mean:.3f}")
            if zone_mean < mean_val - (std_dev / 2):
                zone_alerts.append(f"Zone({i+1},{j+1})")
        zone_means.append(zone_row)

    return {
        "overall_stats": {
            "average_score": f"{mean_val:.4f}",
            "min_max_values": f"{min_val:.4f} / {max_val:.4f}",
            "field_uniformity_std_dev": f"{std_dev:.4f}"
        },
        "health_distribution_percent": {
            "stressed": f"{(stressed_pixels / total_pixels) * 100:.2f}",
            "moderate": f"{(moderate_pixels / total_pixels) * 100:.2f}",
            "healthy": f"{(healthy_pixels / total_pixels) * 100:.2f}"
        },
        "zone_analysis": {
            "average_health_grid": zone_means,
            "potential_problem_zones": zone_alerts if zone_alerts else "None"
        }
    }

# --- 5. API Endpoints ---
@app.post("/predict")
async def predict_disease(crop_type: str = Form(...),
                          file: UploadFile = File(...)):
    crop_type = crop_type.lower()
    if crop_type not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model for crop type '{crop_type}' not found."
        )

    model = loaded_models[crop_type]
    class_names = MODELS_CONFIG[crop_type]["classes"]

    image_bytes = await file.read()
    preprocessed_image = preprocess_image(image_bytes)
    predictions = model.predict(preprocessed_image)

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_label_key = class_names[predicted_index]
    predicted_label_display = (
        predicted_label_key.replace("___", " ").replace("_", " ")
    )

    recommendation = SOLUTION_KNOWLEDGE_BASE.get(
        predicted_label_display,
        {"description": "No specific solution found."}
    )

    return {
        "crop_type": crop_type,
        "predicted_condition": predicted_label_display,
        "confidence": f"{confidence:.2%}",
        "recommendation": recommendation
    }


@app.post("/analyze-live-data")
async def analyze_live_data(latitude: float = Form(...),
                            longitude: float = Form(...)):
    try:
        bbox_size = 0.01
        bbox_coords = [
            longitude - bbox_size / 2,
            latitude - bbox_size / 2,
            longitude + bbox_size / 2,
            latitude + bbox_size / 2
        ]
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)

        evalscript_all_indices = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B03", "B04", "B08", "dataMask"],
                    output: { bands: 3, sampleType: "FLOAT32" }
                };
            }
            function evaluatePixel(sample) {
                let nir = sample.B08;
                let red = sample.B04;
                let green = sample.B03;

                let ndvi = (nir - red) / (nir + red + 0.0001);
                let savi = ((nir - red) / (nir + red + 0.5)) * 1.5;
                let pri = (green - red) / (green + red + 0.0001);

                return [ndvi, savi, pri];
            }
        """

        request = SentinelHubRequest(
            evalscript=evalscript_all_indices,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC"
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            size=(512, 512),
            config=config
        )

        all_indices_data = request.get_data()[0]

        ndvi_map = all_indices_data[:, :, 0]
        savi_map = all_indices_data[:, :, 1]
        pri_map = all_indices_data[:, :, 2]

        # --- Save PNG maps for frontend display ---
        import matplotlib.pyplot as plt

        def save_map_as_png(data_map, filename):
            plt.figure(figsize=(8, 6), dpi=150)
            cmap = plt.cm.RdYlGn
            data_map = np.clip(data_map, 0, 1)
            img = plt.imshow(data_map, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(img, fraction=0.046, pad=0.04, label="Index Value")
            plt.axis("off")
            save_path = os.path.join(MATLAB_MAPS_FOLDER, filename)
            os.makedirs(MATLAB_MAPS_FOLDER, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight",
                        pad_inches=0.1, dpi=150)
            plt.close()
            return save_path

        save_map_as_png(ndvi_map, "ndvi_map.png")
        save_map_as_png(savi_map, "savi_map.png")
        save_map_as_png(pri_map, "pri_map.png")

        return {
            "ndvi_summary": generate_health_report("NDVI", ndvi_map),
            "savi_summary": generate_health_report("SAVI", savi_map),
            "pri_summary": generate_health_report("PRI", pri_map)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching data: {str(e)}"
        )


@app.get("/health-map/{map_type}")
async def get_health_map(map_type: str):
    map_type = map_type.lower()
    if map_type not in ["ndvi", "savi", "pri"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid map type. Use: ndvi, savi, pri"
        )

    for ext in ["png", "jpg", "jpeg"]:
        file_path = os.path.join(
            MATLAB_MAPS_FOLDER, f"{map_type}_map.{ext}"
        )
        if os.path.exists(file_path):
            return FileResponse(file_path)

    raise HTTPException(
        status_code=404,
        detail=f"Map file not found for '{map_type}' in Matlab_Output folder."
    )
