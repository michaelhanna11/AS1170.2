import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import io
from datetime import datetime
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from io import BytesIO
from PIL import Image as PILImage

# Program details
PROGRAM_VERSION = "1.0 - 2025"
PROGRAM = "Wind Load Calculator to AS/NZS 1170.2:2021"
COMPANY_NAME = "tekhne Consulting Engineers"
COMPANY_ADDRESS = "   "  # Placeholder; update with actual address if needed
LOGO_URL = "https://drive.google.com/uc?export=download&id=1VebdT2loVGX57noP9t2GgQhwCNn8AA3h"
FALLBACK_LOGO_URL = "https://onedrive.live.com/download?cid=A48CC9068E3FACE0&resid=A48CC9068E3FACE0%21s252b6fb7fcd04f53968b2a09114d33ed"

def load_logo(url):
    try:
        response = requests.get(url, timeout=5)
        img = PILImage.open(BytesIO(response.content))
        return img
    except:
        return None

# Load logo (try primary URL first, then fallback)
logo = load_logo(LOGO_URL) or load_logo(FALLBACK_LOGO_URL)

# WindLoadCalculator class (unchanged)
class WindLoadCalculator:
    # [Previous WindLoadCalculator class implementation remains exactly the same]
    # ... (include all the WindLoadCalculator methods exactly as in your original code)

# PDF generation functions (unchanged except for adding construction period to input table)
def build_elements(inputs, results, project_number, project_name):
    # [Previous build_elements implementation remains exactly the same]
    # ... (include all the PDF generation code exactly as in your original code)

def generate_pdf_report(inputs, results, project_number, project_name):
    # [Previous generate_pdf_report implementation remains exactly the same]
    # ... (include all the PDF generation code exactly as in your original code)

# Streamlit UI - This is the only main() function now
def main():
    # Set page configuration with a title for the browser tab
    st.set_page_config(page_title="Wind Load Calculator - AS/NZS 1170.2:2021")

    # Logo display
    if logo:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(logo, width=200)
        st.markdown('</div>', unsafe_allow_html=True)

    st.title("Wind Load Calculator (AS/NZS 1170.2:2021)")
    calculator = WindLoadCalculator()

    with st.form(key='wind_load_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            # Project details
            project_number = st.text_input("Project Number", value="PRJ-001")
            
        with col2:
            project_name = st.text_input("Project Name", value="Sample Project")
        
        # Construction Duration input - NOW VISIBLE
        construction_period = st.selectbox(
            "Construction Duration",
            ["1 week", "1 month", "6 months", "More than 6 months"],
            index=2  # Default to 6 months
        )
        st.markdown("---")  # Horizontal line for visual separation

        # Location
        st.subheader("Location")
        location = st.selectbox("Select Location", calculator.valid_locations, 
                              index=calculator.valid_locations.index("Sydney"))

        # [Rest of your form inputs remain exactly the same...]
        # Importance Level
        importance_level = st.selectbox("Importance Level for ULS", ["I", "II", "III"])

        # Terrain Category
        st.subheader("Terrain Category")
        terrain_options = {f"{key} ({value['name']}): {value['desc']}": key for key, value in calculator.terrain_categories.items()}
        terrain_choice = st.selectbox("Select Terrain Category", list(terrain_options.keys()))
        terrain_category = calculator.terrain_categories[terrain_options[terrain_choice]]["name"]

        # Reference Height
        reference_height = st.number_input("Reference Height (m)", min_value=0.1, value=10.0, step=0.1)

        # Region-specific inputs
        region = calculator.determine_wind_region(location)
        distance_from_coast_km = None
        if region in ["C", "D"]:
            distance_from_coast_km = st.number_input("Distance from Coast (km)", min_value=50.0, max_value=200.0, value=50.0, step=1.0)

        # Structure Type
        st.subheader("Structure Type")
        structure_choice = st.selectbox("Select Structure Type", list(calculator.structure_types.values()))
        structure_type = structure_choice

        # Structure-specific inputs
        b = c = user_C_shp = None
        has_return_corner = False
        if structure_type == "Free Standing Wall":
            b = st.number_input("Width of the Wall (b, m)", min_value=0.1, value=10.0, step=0.1)
            c = st.number_input("Height of the Wall (c, m)", min_value=0.1, max_value=reference_height, value=min(3.0, reference_height), step=0.1)
            one_c = c
            st.write(f"Note: 1c = {one_c:.2f} m (based on wall height c)")
            has_return_corner = st.checkbox(f"Return Corner Extends More Than 1c ({one_c:.2f} m)")
        elif structure_type == "Protection Screens":
            user_C_shp = st.number_input("Aerodynamic Shape Factor (C_shp)", min_value=0.1, value=1.0, step=0.01)

        submit_button = st.form_submit_button(label="Calculate and Generate Report")

    if submit_button:
        h = reference_height
        inputs = {
            'location': location,
            'region': region,
            'importance_level': importance_level,
            'terrain_category': terrain_category,
            'reference_height': reference_height,
            'distance_from_coast_km': distance_from_coast_km,
            'structure_type': structure_type,
            'b': b,
            'c': c,
            'has_return_corner': has_return_corner,
            'user_C_shp': user_C_shp,
            'construction_period': construction_period  # Now included in inputs
        }

        # [Rest of your calculation and PDF generation code remains exactly the same...]

if __name__ == "__main__":
    main()
