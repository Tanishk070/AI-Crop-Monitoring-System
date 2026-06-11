import os
import streamlit as st
import requests
from PIL import Image
import datetime
import random
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="FASALVision | SIH 2025",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"

# --- Paths ---
# PROJECT_ROOT = .../AI-Crop-Monitoring-System
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logo_path = os.path.join(PROJECT_ROOT, "assets", "FasalVision_Logo.png")

# --- Sidebar ---
with st.sidebar:
    st.image(logo_path, use_container_width=True)
    st.title("🌿 FASALVision")
    page = st.radio(
        "Choose a feature",
        [
            "🔬 Leaf Diagnosis",
            "🛰️ Field Health Dashboard",
            "📈 Yield Prediction",
            "🌦️ Weather Forecast",
            "🌱 Soil Sensor Dashboard",
            "📊 Temporal Trends",
            "🤖 AI Chatbot (Preview)"
        ],
        label_visibility="hidden"
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Prototype developed for **Smart India Hackathon 2025**.\n\n"
        "AI-powered decision support system for smarter and sustainable farming."
    )

# =====================================================================================
# --- Page 1: Leaf Diagnosis ---
# =====================================================================================
if page == "🔬 Leaf Diagnosis":
    st.title("🔬 Leaf-Level Disease Diagnosis")
    st.markdown("Upload a leaf image to get an instant AI-powered diagnosis and recommended solution.")

    with st.container(border=True):
        crop_type = st.selectbox(
            "1. Select Crop Type",
            options=["rice", "tomato", "potato", "wheat"]
        )
        uploaded_file = st.file_uploader(
            "2. Upload a Leaf Image",
            type=["jpg", "png", "jpeg"]
        )
        diagnose_button = st.button("Diagnose Leaf Condition", type="primary", use_container_width=True)

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded {crop_type.capitalize()} Leaf", use_container_width=True)
        with col2:
            if diagnose_button:
                with st.spinner('Analyzing... Our AI expert is on the case!'):
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    payload = {'crop_type': crop_type}
                    try:
                        response = requests.post(f"{API_URL}/predict", files=files, data=payload)
                        if response.status_code == 200:
                            result = response.json()
                            st.success("**Diagnosis Complete!**")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("Predicted Condition")
                                st.markdown(f"<h4>{result['predicted_condition']}</h4>", unsafe_allow_html=True)
                            with c2:
                                confidence_percent = float(result['confidence'].strip('%'))
                                st.progress(int(confidence_percent), text=f"Confidence Score: {result['confidence']}")
                            with st.expander("Show Recommended Actions", expanded=True):
                                rec = result.get('recommendation', {})
                                for key, value in rec.items():
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            st.markdown("---")
                            st.subheader("Need a Second Opinion?")
                            st.markdown("Submit your case for a personalized review by one of our agricultural experts.")
                            if st.button("Request Expert Review", use_container_width=True):
                                st.success("Your case has been submitted!")
                        else:
                            st.error(f"Error from server: {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Could not connect to the backend.")
            else:
                st.info("Click 'Diagnose' to see the results.")

# =====================================================================================
# --- Page 2: Field Health Dashboard ---
# =====================================================================================
elif page == "🛰️ Field Health Dashboard":
    st.title("🛰️ Field-Wide Health Dashboard")
    st.markdown("Get a live satellite-based analysis of your field's condition using NDVI, SAVI, and PRI indices.")

    from streamlit_folium import st_folium
    from streamlit_js_eval import get_geolocation
    import folium
    from geopy.geocoders import Nominatim

    st.subheader("Select Your Field Location")
    option = st.radio(
        "Choose how you want to select your location:",
        ["📍 Use My Current Location", "🗺️ Select on Map", "🔍 Search by Area Name"],
        horizontal=True
    )

    latitude, longitude = None, None

    if option == "📍 Use My Current Location":
        loc = get_geolocation()
        if loc:
            latitude = loc["coords"]["latitude"]
            longitude = loc["coords"]["longitude"]
            st.success(f"✅ Detected Location: {latitude:.4f}, {longitude:.4f}")
        else:
            st.warning("Please allow browser location access.")

    elif option == "🗺️ Select on Map":
        st.markdown("Click on the map to mark your field.")
        m = folium.Map(location=[23.1667, 79.9333], zoom_start=5)
        m.add_child(folium.LatLngPopup())
        output = st_folium(m, width=700, height=500)
        if output and "last_clicked" in output and output["last_clicked"]:
            latitude = output["last_clicked"]["lat"]
            longitude = output["last_clicked"]["lng"]
            st.success(f"✅ Selected Location: {latitude:.4f}, {longitude:.4f}")

    elif option == "🔍 Search by Area Name":
        place = st.text_input("Enter Area/City/Village Name:")
        if st.button("Search"):
            geolocator = Nominatim(user_agent="fasalvision")
            location = geolocator.geocode(place)
            if location:
                latitude, longitude = location.latitude, location.longitude
                st.success(f"✅ Found Location: {place} ({latitude:.4f}, {longitude:.4f})")
            else:
                st.error("❌ Location not found. Try again.")

    def generate_summary(index_name, avg_value):
        if index_name == "ndvi":
            if avg_value < 0.3:
                return "Vegetation health is poor — possible water stress."
            elif avg_value < 0.6:
                return "Moderate vegetation activity."
            else:
                return "High vegetation activity — crops are healthy."
        elif index_name == "savi":
            if avg_value < 0.3:
                return "Low soil-adjusted vegetation index — possibly sparse crops."
            elif avg_value < 0.6:
                return "Moderate SAVI — balanced soil and crop conditions."
            else:
                return "Healthy vegetation cover detected."
        elif index_name == "pri":
            if avg_value < 0.2:
                return "Low photosynthetic activity — possible stress."
            elif avg_value < 0.5:
                return "Moderate photosynthetic activity observed."
            else:
                return "Strong photosynthetic performance."

    if latitude and longitude:
        st.markdown("---")
        if st.button("🚀 Fetch & Analyze Live Satellite Data", type="primary", use_container_width=True):
            with st.spinner('Analyzing satellite data...'):
                try:
                    payload = {'latitude': float(latitude), 'longitude': float(longitude)}
                    response = requests.post(f"{API_URL}/analyze-live-data", data=payload)

                    if response.status_code == 200:
                        st.success("✅ Live analysis complete!")
                        live_data = response.json()

                        st.subheader("🌾 Live Field Health Report")
                        for index_name in ["ndvi", "savi", "pri"]:
                            summary = live_data.get(f"{index_name}_summary")
                            if summary:
                                with st.expander(f"📊 {index_name.upper()} Report", expanded=(index_name == 'ndvi')):
                                    stats = summary.get('overall_stats', {})
                                    dist = summary.get('health_distribution_percent', {})
                                    zones = summary.get('zone_analysis', {})

                                    avg_score = float(stats.get('average_score', 0.0))
                                    st.metric(f"Average {index_name.upper()}", value=stats.get('average_score', 'N/A'))
                                    st.write(f"**Min / Max Values:** {stats.get('min_max_values', 'N/A')}")
                                    st.write(f"**Field Uniformity (Std Dev):** {stats.get('field_uniformity_std_dev', 'N/A')}")
                                    st.markdown("---")
                                    st.write(
                                        f"🌱 Healthy: {dist.get('healthy', 0)}% | "
                                        f"😐 Moderate: {dist.get('moderate', 0)}% | "
                                        f"⚠️ Stressed: {dist.get('stressed', 0)}%"
                                    )

                                    alerts = zones.get('potential_problem_zones', [])
                                    if alerts != "None" and alerts:
                                        st.warning(f"⚠️ Problem Zones: {', '.join(alerts)}")
                                    else:
                                        st.success("✅ No significant problem zones detected.")

                                    map_url = f"{API_URL}/health-map/{index_name}"
                                    st.image(map_url, caption=f"{index_name.upper()} Map", use_container_width=True)
                                    summary_text = generate_summary(index_name, avg_score)
                                    st.markdown(f"🧠 **Summary:** {summary_text}")
                    else:
                        # 🔹 NEW: show actual backend error detail
                        try:
                            err = response.json()
                            detail = err.get("detail", err)
                            st.error(f"Error fetching live data: {detail}")
                        except Exception:
                            st.error(f"Error fetching live data. Status code: {response.status_code}")
                except Exception:
                    st.error("🚫 Backend connection error.")
    else:
        st.info("Please select or detect your field location first.")

# =====================================================================================
# --- Page 3: Yield Prediction (Future Scope) ---
# =====================================================================================
elif page == "📈 Yield Prediction":
    st.title("📈 Yield Prediction (AI + LSTM - Demo)")
    st.info("Simulated prototype for AI-powered yield forecasting using soil, weather, and vegetation data.")

    st.subheader("Input Farm Data (Demo)")
    with st.form("yield_form"):
        col1, col2 = st.columns(2)
        with col1:
            crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Tomato", "Potato"])
            soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 60)
            humidity = st.slider("Humidity (%)", 0, 100, 70)
        with col2:
            temperature = st.slider("Average Temperature (°C)", 10, 45, 30)
            ndvi = st.slider("Average NDVI Value", 0.0, 1.0, 0.6)
            savi = st.slider("Average SAVI Value", 0.0, 1.0, 0.5)
        submitted = st.form_submit_button("Predict Yield")

    if submitted:
        st.success("✅ AI model simulated prediction complete.")
        predicted_yield = round(random.uniform(2.5, 5.0), 2)
        st.metric("Predicted Yield", f"{predicted_yield} tons/hectare")

        st.markdown("📊 **Simulated Yield Trend:**")
        x = list(range(1, 6))
        y = [predicted_yield * random.uniform(0.9, 1.1) for _ in x]
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='o')
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Predicted Yield (t/ha)")
        st.pyplot(fig)

    st.info("🔬 Model training under development — powered by LSTM for temporal prediction.")

# =====================================================================================
# --- Page 4: Weather Forecast (Demo) ---
# =====================================================================================
elif page == "🌦️ Weather Forecast":
    st.title("🌦️ Weather Forecast (Demo)")
    st.info("Displays current and predicted weather data using OpenWeatherMap API (demo mode).")

    st.metric("Temperature", "31°C")
    st.metric("Humidity", "68%")
    st.metric("Rainfall Chance", "25%")
    st.metric("Wind Speed", "12 km/h")
    st.success("☀️ Weather conditions favorable for healthy crop growth.")

# =====================================================================================
# --- Page 5: Soil Sensor Dashboard (IoT Mock) ---
# =====================================================================================
elif page == "🌱 Soil Sensor Dashboard":
    st.title("🌱 IoT Soil Sensor Dashboard (Demo)")
    st.info("Displays simulated soil parameters from smart sensors.")

    st.metric("Soil pH", "6.8")
    st.metric("Moisture", "72%")
    st.metric("Nitrogen", "Optimal")
    st.metric("Phosphorus", "Adequate")
    st.metric("Potassium", "Sufficient")
    st.success("✅ Soil parameters are optimal for most crops.")

# =====================================================================================
# --- Page 6: Temporal Trends (Demo) ---
# =====================================================================================
elif page == "📊 Temporal Trends":
    st.title("📊 Temporal Trend Analyzer (Demo)")
    st.info("Visualizing NDVI, SAVI, and PRI changes over time (sample data).")

    x = [f"Day {i}" for i in range(1, 11)]
    ndvi = [round(random.uniform(0.4, 0.8), 2) for _ in x]
    savi = [round(random.uniform(0.3, 0.7), 2) for _ in x]
    pri = [round(random.uniform(0.2, 0.6), 2) for _ in x]

    fig, ax = plt.subplots()
    ax.plot(x, ndvi, label="NDVI", marker='o')
    ax.plot(x, savi, label="SAVI", marker='s')
    ax.plot(x, pri, label="PRI", marker='^')
    ax.set_ylabel("Index Value")
    ax.set_title("Vegetation Indices Over Time (Simulated)")
    ax.legend()
    st.pyplot(fig)
    st.success("📈 These patterns will help identify early stress or disease signs once live data is integrated.")

# =====================================================================================
# --- Page 7: AI Chatbot (Preview) ---
# =====================================================================================
elif page == "🤖 AI Chatbot (Preview)":
    st.title("🤖 FASALBot — Your Smart Agriculture Assistant (Coming Soon)")
    st.info("Conversational AI assistant to answer farmers' queries and guide in real-time.")

    st.chat_message("assistant").write("👋 Hello! I'm FASALBot. Ask me anything about your crops...")
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write("🤖 This feature is under development. Stay tuned for live chat support!")

    st.warning("🚧 AI Chatbot integration with OpenAI/Local LLM in progress.")
