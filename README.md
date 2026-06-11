# 🌿 FASALVision

An integrated AI-powered platform for precision agriculture, developed for the Smart India Hackathon 2025. FASALVision combines leaf-level disease diagnosis with field-level satellite data analysis to provide farmers with comprehensive and actionable insights.

---

## 📜 About The Project

Agriculture faces major challenges from soil degradation, pest outbreaks, and unpredictable weather, leading to significant yield loss. Traditional monitoring methods are often slow and lack precision. FASALVision addresses this by leveraging the power of AI and multispectral imaging to provide a dual-layered analysis:

* **Micro-Level Diagnosis:** Farmers can upload a photo of a single plant leaf to get an instant, highly accurate disease diagnosis from one of our four specialized AI models.
* **Macro-Level Monitoring:** The platform processes Sentinel-2 satellite imagery using MATLAB to generate field-wide health maps (NDVI, SAVI, PRI), identifying stress zones before they become critical.

This hybrid approach provides a complete picture of crop health, enabling proactive and sustainable farm management.

**(You can add a screenshot of your dashboard here)**
`![FASALVision Dashboard](./assets/dashboard_screenshot.png)`

---

## ✨ Key Features

* **🔬 Multi-Crop Disease Diagnosis:** Utilizes four specialized AI models for Rice, Tomato, Potato, and Wheat.
* **💡 Actionable Solutions:** Provides practical recommendations for treatment and prevention for each diagnosed disease.
* **🛰️ Field-Level Health Monitoring:** Generates visual health maps (NDVI, SAVI, PRI) from satellite data.
* **📊 Detailed Numerical Analysis:** Offers a comprehensive report on field health, including stress distribution and identification of problem zones.
* **🤝 Expert Consultation:** A "Human-in-the-Loop" feature allows farmers to request a review from an agricultural expert.
* **🔮 Future-Ready:** Designed with a clear roadmap for automation, including direct integration with the Copernicus API and a yield prediction engine.

---

## 🛠️ Built With

This project integrates a diverse set of modern technologies to deliver a robust solution.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **AI Models** | Python, TensorFlow, Keras | Training and inference for four specialized CNNs. |
| **Spectral Analysis**| MATLAB, Image Processing Toolbox| Processing satellite imagery to generate vegetation indices. |
| **Backend API** | Python, FastAPI | Serving all AI models, MATLAB data, and maps via REST endpoints. |
| **Frontend** | Python, Streamlit | Creating the interactive user dashboard. |
| **Data Handling** | Pandas, NumPy, Scipy | Data manipulation and reading `.mat` files. |
| **Datasets** | PlantVillage, Sentinel-2 | Sourcing data for model training and field analysis. |

---

## 🚀 Getting Started

To get a local copy up and running, follow these steps.

### **Prerequisites**

1.  Clone the repository:
    ```sh
    git clone [https://github.com/Tanishk070/AI-Crop-Monitoring-System.git](https://github.com/Tanishk070/AI-Crop-Monitoring-System.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd AI-Crop-Monitoring-System
    ```
3.  Create and activate a Python virtual environment:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install all required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### **Running the Application**

This project consists of two components that must be run simultaneously in **two separate terminals**.

**1. Start the Backend Server**

* In your first terminal (with the virtual environment activated), run the following command from the project's root directory:
    ```sh
    uvicorn python.api.app:app --reload
    ```
* The server will start on `http://127.0.0.1:8000`.

**2. Start the Frontend Dashboard**

* Open a **new, second terminal**.
* Navigate to the project's root directory and activate the virtual environment again.
* Run the following command:
    ```sh
    streamlit run python/dashboard/dashboard.py
    ```
* A new tab will automatically open in your web browser with the FASALVision dashboard.

---

## 🏛️ Architecture

The system uses a dual-pipeline architecture to process both on-the-ground and satellite data, which are then unified in the backend.

**(You can add your System Architecture diagram here)**
`![System Architecture](./assets/FASALVision System Architecture.png)`

---
