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

# Program details
PROGRAM_VERSION = "1.0 - 2025"
PROGRAM = "Wind Load Calculator to AS/NZS 1170.2:2021"
COMPANY_NAME = "tekhne Consulting Engineers"
COMPANY_ADDRESS = "   "  # Placeholder; update with actual address if needed
LOGO_URL = "https://drive.google.com/uc?export=download&id=1VebdT2loVGX57noP9t2GgQhwCNn8AA3h"
FALLBACK_LOGO_URL = "https://onedrive.live.com/download?cid=A48CC9068E3FACE0&resid=A48CC9068E3FACE0%21s252b6fb7fcd04f53968b2a09114d33ed"




# WindLoadCalculator class (unchanged)
class WindLoadCalculator:
    def __init__(self):
        self.V_R_table = {
            "A0": {25: 37, 100: 41, 250: 43},
            "A1": {25: 37, 100: 41, 250: 43},
            "A2": {25: 37, 100: 41, 250: 43},
            "A3": {25: 37, 100: 41, 250: 43},
            "A4": {25: 37, 100: 41, 250: 43},
            "A5": {25: 37, 100: 41, 250: 43},
            "B1": {25: 39, 100: 48, 250: 53},
            "B2": {25: 39, 100: 48, 250: 53},
            "C": {25: 47, 100: 56, 250: 62},
            "D": {25: 53, 100: 66, 250: 74},
        }
        self.M_c_table = {
            "A0": 1.0, "A1": 1.0, "A2": 1.0, "A3": 1.0, "A4": 1.0, "A5": 1.0,
            "B1": 1.0, "B2": 1.0, "C": 1.05, "D": 1.05,
        }
        self.M_z_cat_table = {
            "TC1": {3: 0.97, 5: 1.01, 10: 1.08, 15: 1.12, 20: 1.14, 30: 1.18, 40: 1.21, 50: 1.23, 75: 1.27, 100: 1.31, 150: 1.36, 200: 1.39},
            "TC2": {3: 0.91, 5: 0.91, 10: 1.00, 15: 1.05, 20: 1.08, 30: 1.12, 40: 1.16, 50: 1.18, 75: 1.22, 100: 1.24, 150: 1.27, 200: 1.29},
            "TC2.5": {3: 0.87, 5: 0.87, 10: 0.92, 15: 0.97, 20: 1.01, 30: 1.06, 40: 1.10, 50: 1.13, 75: 1.17, 100: 1.20, 150: 1.24, 200: 1.27},
            "TC3": {3: 0.83, 5: 0.83, 10: 0.83, 15: 0.89, 20: 0.94, 30: 1.00, 40: 1.04, 50: 1.07, 75: 1.12, 100: 1.16, 150: 1.21, 200: 1.24},
            "TC4": {3: 0.75, 5: 0.75, 10: 0.75, 15: 0.75, 20: 0.75, 30: 0.80, 40: 0.85, 50: 0.90, 75: 0.98, 100: 1.03, 150: 1.11, 200: 1.16},
        }
        self.regions_with_interpolation = ["C", "D"]
        self.terrain_categories = {
            "1": {"name": "TC1", "desc": "Exposed open terrain, few/no obstructions (e.g., open ocean, flat plains)"},
            "2": {"name": "TC2", "desc": "Open terrain, grassland, few obstructions 1.5m-5m (e.g., farmland)"},
            "2.5": {"name": "TC2.5", "desc": "Developing outer urban, some trees, 2-10 buildings/ha"},
            "3": {"name": "TC3", "desc": "Suburban, many obstructions 3m-10m, ≥10 houses/ha (e.g., housing estates)"},
            "4": {"name": "TC4", "desc": "City centers, large/high (10m-30m) closely spaced buildings (e.g., industrial complexes)"},
        }
        self.structure_types = {
            "1": "Free Standing Wall",
            "2": "Circular Tank",
            "3": "Attached Canopy",
            "4": "Protection Screens",
        }
        self.valid_locations = [
            "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
            "Darwin", "Cairns", "Townsville", "Port Hedland", "Alice Springs", "Hobart"
        ]

    def determine_wind_region(self, location):
        region_map = {
            "Sydney": "A2", "Melbourne": "A4", "Brisbane": "B1", "Perth": "A1", "Adelaide": "A5",
            "Darwin": "C", "Cairns": "C", "Townsville": "B2", "Port Hedland": "D", "Alice Springs": "A0", "Hobart": "A4",
        }
        return region_map.get(location, "A2")

    def interpolate_V_R(self, region, R, distance_from_coast_km):
        if region not in self.regions_with_interpolation:
            return self.V_R_table[region][R]
        V_R_50km = self.V_R_table[region][R]
        V_R_100km = V_R_50km * 0.95
        V_R_200km = V_R_50km * 0.90
        if distance_from_coast_km <= 50:
            return V_R_50km
        elif distance_from_coast_km <= 100:
            fraction = (distance_from_coast_km - 50) / (100 - 50)
            return V_R_50km + fraction * (V_R_100km - V_R_50km)
        elif distance_from_coast_km <= 200:
            fraction = (distance_from_coast_km - 100) / (200 - 100)
            return V_R_100km + fraction * (V_R_200km - V_R_100km)
        else:
            return V_R_200km

    def determine_V_R(self, region, limit_state, importance_level=None, distance_from_coast_km=None):
        if limit_state == "SLS":
            return 16.0
        R_map = {"I": 25, "II": 100, "III": 250}
        if importance_level not in R_map:
            raise ValueError("Importance level must be 'I', 'II', or 'III' for ULS.")
        R = R_map[importance_level]
        if region in self.regions_with_interpolation and distance_from_coast_km is not None:
            return self.interpolate_V_R(region, R, distance_from_coast_km)
        return self.V_R_table[region][R]

    def determine_M_c(self, region):
        return self.M_c_table[region]

    def determine_M_d(self, region):
        return 1.0

    def determine_M_s(self, region):
        return 1.0

    def determine_M_t(self, region):
        return 1.0

    def determine_M_z_cat(self, region, terrain_category, height):
        if region == "A0" and height > 100:
            return 1.24 if height <= 200 else 1.24
        terrain_data = self.M_z_cat_table[terrain_category]
        heights = sorted(terrain_data.keys())
        if height in heights:
            return terrain_data[height]
        if height <= heights[0]:
            return terrain_data[heights[0]]
        if height >= heights[-1]:
            return terrain_data[heights[-1]]
        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i + 1]
            if h1 < height <= h2:
                m1, m2 = terrain_data[h1], terrain_data[h2]
                fraction = (height - h1) / (h2 - h1)
                return m1 + fraction * (m2 - m1)
        return 1.0

    def calculate_site_wind_speed(self, V_R, M_d, M_c, M_s, M_t, M_z_cat):
        return V_R * M_d * M_c * M_s * M_t * M_z_cat

    def calculate_design_wind_speed(self, V_sit_beta, limit_state):
        if limit_state == "ULS":
            return max(V_sit_beta, 30.0)
        return V_sit_beta

    def calculate_Cpn_freestanding_wall(self, b, c, h, theta, distance_from_windward_end=None, has_return_corner=False):
        b_over_c = b / c
        c_over_h = c / h
        if theta == 0:
            if 0.5 <= b_over_c <= 5:
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else:
                    Cpn = 1.4 + 0.3 * log10(b_over_c)
            else:
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.7 - 0.5 * c_over_h
                else:
                    Cpn = 1.4 + 0.3 * log10(b_over_c)
            e = 0.0
        elif theta == 45:
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=45°.")
            if 0.5 <= b_over_c <= 5:
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else:
                    Cpn = 1.4 + 0.3 * log10(b_over_c)
            else:
                if c_over_h <= 0.7:
                    if distance_from_windward_end <= 2 * c:
                        Cpn = 3.0
                    elif distance_from_windward_end <= 4 * c:
                        Cpn = 1.5
                    else:
                        Cpn = 0.75
                else:
                    if distance_from_windward_end <= 2 * h:
                        Cpn = 2.4
                    elif distance_from_windward_end <= 4 * h:
                        Cpn = 1.2
                    else:
                        Cpn = 0.6
                if has_return_corner:
                    if distance_from_windward_end <= 2 * c:
                        Cpn = 2.2
                    if distance_from_windward_end <= 2 * h:
                        Cpn = 1.8
            e = 0.2 * b
        elif theta == 90:
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=90°.")
            if c_over_h <= 0.7:
                if distance_from_windward_end <= 2 * c:
                    Cpn = 1.2
                elif distance_from_windward_end <= 4 * c:
                    Cpn = 0.6
                else:
                    Cpn = 0.3
            else:
                if distance_from_windward_end <= 2 * h:
                    Cpn = 1.0
                elif distance_from_windward_end <= 4 * h:
                    Cpn = 0.25
                else:
                    Cpn = 0.25
            Cpn = abs(Cpn)
            e = 0.0
        else:
            raise ValueError("Theta must be 0°, 45°, or 90°.")
        return Cpn, e

    def calculate_aerodynamic_shape_factor(self, structure_type, user_C_shp=None, b=None, c=None, h=None, theta=None, distance_from_windward_end=None, has_return_corner=False):
        K_p = 1.0
        if structure_type == "Free Standing Wall":
            Cpn, e = self.calculate_Cpn_freestanding_wall(b, c, h, theta, distance_from_windward_end, has_return_corner)
            return Cpn * K_p, e
        elif structure_type == "Circular Tank":
            return 0.8, 0.0
        elif structure_type == "Attached Canopy":
            return 1.2, 0.0
        elif structure_type == "Protection Screens":
            if user_C_shp is None:
                raise ValueError("C_shp required for Protection Screens.")
            return user_C_shp, 0.0
        else:
            raise ValueError("Invalid structure type.")

    def calculate_wind_pressure(self, V_des_theta, C_shp):
        rho_air = 1.2
        C_dyn = 1.0
        return (0.5 * rho_air) * (V_des_theta ** 2) * C_shp * C_dyn / 1000

    def calculate_pressure_distribution(self, b, c, h, V_des_theta, theta, has_return_corner=False):
        num_points = 100
        distances = np.linspace(0, b, num_points)
        pressures = []
        for d in distances:
            C_shp, _ = self.calculate_aerodynamic_shape_factor(
                "Free Standing Wall", None, b, c, h, theta, distance_from_windward_end=d, has_return_corner=has_return_corner
            )
            p = self.calculate_wind_pressure(V_des_theta, C_shp)
            pressures.append(p)
        return distances, pressures

    def calculate_pressure_vs_height(self, region, terrain_category, reference_height, limit_state, importance_level, distance_from_coast_km, C_shp):
        height_step = 5.0
        heights = np.arange(0, reference_height + height_step, height_step)
        if heights[-1] > reference_height:
            heights = heights[:-1]
        heights = np.append(heights, reference_height)
        heights = np.sort(heights)
        V_des_values = []
        pressures = []
        V_R = self.determine_V_R(region, limit_state, importance_level, distance_from_coast_km)
        M_d = self.determine_M_d(region)
        M_c = self.determine_M_c(region)
        M_s = self.determine_M_s(region)
        M_t = self.determine_M_t(region)
        for h in heights:
            M_z_cat = self.determine_M_z_cat(region, terrain_category, h)
            V_sit_beta = self.calculate_site_wind_speed(V_R, M_d, M_c, M_s, M_t, M_z_cat)
            V_des = self.calculate_design_wind_speed(V_sit_beta, limit_state)
            p = self.calculate_wind_pressure(V_des, C_shp)
            V_des_values.append(V_des)
            pressures.append(p)
        return heights, V_des_values, pressures

# PDF generation functions (unchanged)
def build_elements(inputs, results, project_number, project_name):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=16, spaceAfter=6, alignment=1)
    subtitle_style = ParagraphStyle(name='SubtitleStyle', parent=styles['Normal'], fontSize=10, spaceAfter=6, alignment=1)
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=12, spaceAfter=4)
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    justified_style = ParagraphStyle(name='JustifiedStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4, alignment=TA_JUSTIFY)
    table_header_style = ParagraphStyle(name='TableHeaderStyle', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_LEFT)
    table_cell_style = ParagraphStyle(name='TableCellStyle', parent=styles['Normal'], fontSize=8, alignment=TA_LEFT, leading=9)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ])
    input_table_cell_style = ParagraphStyle(name='InputTableCellStyle', parent=styles['Normal'], fontSize=8, alignment=TA_LEFT, leading=9)
    input_table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ])
    elements = []

    logo_file = "logo.png"
    try:
        response = requests.get(LOGO_URL, stream=True, timeout=10)
        response.raise_for_status()
        with open(logo_file, 'wb') as f:
            f.write(response.content)
    except Exception:
        try:
            response = requests.get(FALLBACK_LOGO_URL, stream=True, timeout=10)
            response.raise_for_status()
            with open(logo_file, 'wb') as f:
                f.write(response.content)
        except Exception:
            logo_file = None

    company_text = f"<b>{COMPANY_NAME}</b><br/>{COMPANY_ADDRESS}"
    company_paragraph = Paragraph(company_text, normal_style)
    logo = Image(logo_file, width=50*mm, height=20*mm) if logo_file else Paragraph("[Logo Placeholder]", normal_style)
    header_data = [[logo, company_paragraph]]
    header_table = Table(header_data, colWidths=[60*mm, 120*mm])
    header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('ALIGN', (1, 0), (1, 0), 'CENTER')]))
    elements.append(header_table)
    elements.append(Spacer(1, 6*mm))

    elements.append(Paragraph("Wind Load Report for Structural Design to AS/NZS 1170.2:2021", title_style))
    project_details = f"Project Number: {project_number}<br/>Project Name: {project_name}<br/>Date: {datetime.now().strftime('%B %d, %Y')}"
    elements.append(Paragraph(project_details, subtitle_style))
    elements.append(Spacer(1, 4*mm))

    elements.append(Paragraph("Introduction", heading_style))
    structure_type = inputs['structure_type']
    if structure_type == "Free Standing Wall":
        intro_text = (
            "This report presents the wind load calculations for structural design as per AS/NZS 1170.2:2021, "
            "specifically following the guidelines for determining regional, site, and design wind speeds, as well as "
            "wind pressures for a free standing wall. The Australian/New Zealand Standard AS/NZS 1170.2 provides "
            "methodologies for calculating wind actions on structures, ensuring safety and structural integrity under "
            "various wind conditions. This report documents the wind pressures for Ultimate Limit State (ULS) and "
            "Serviceability Limit State (SLS) across three wind directions (θ = 0°, 45°, and 90°). The results are "
            "presented for a free standing wall, considering factors such as terrain category, wind region, and the "
            "presence of a return corner. The aerodynamic shape factors are determined based on Tables B.2(A) to B.2(D), "
            "with the footer condition applied where a return corner extends more than 1c."
        )
    elif structure_type == "Circular Tank":
        intro_text = (
            "This report presents the wind load calculations for structural design as per AS/NZS 1170.2:2021, "
            "specifically following the guidelines for determining regional, site, and design wind speeds, as well as "
            "wind pressures for a circular tank. The Australian/New Zealand Standard AS/NZS 1170.2 provides "
            "methodologies for calculating wind actions on structures, ensuring safety and structural integrity under "
            "various wind conditions. This report documents the wind pressures for Ultimate Limit State (ULS) and "
            "Serviceability Limit State (SLS). The results are presented for a circular tank, considering factors "
            "such as terrain category and wind region. The aerodynamic shape factors are determined based on relevant "
            "tables in AS/NZS 1170.2, such as Table 5.3(A) for circular sections."
        )
    elif structure_type == "Attached Canopy":
        intro_text = (
            "This report presents the wind load calculations for structural design as per AS/NZS 1170.2:2021, "
            "specifically following the guidelines for determining regional, site, and design wind speeds, as well as "
            "wind pressures for an attached canopy. The Australian/New Zealand Standard AS/NZS 1170.2 provides "
            "methodologies for calculating wind actions on structures, ensuring safety and structural integrity under "
            "various wind conditions. This report documents the wind pressures for Ultimate Limit State (ULS) and "
            "Serviceability Limit State (SLS). The results are presented for an attached canopy, considering factors "
            "such as terrain category and wind region. The aerodynamic shape factors are determined based on relevant "
            "tables in AS/NZS 1170.2, such as Table 7.2 for canopies."
        )
    elif structure_type == "Protection Screens":
        intro_text = (
            "This report presents the wind load calculations for structural design as per AS/NZS 1170.2:2021, "
            "specifically following the guidelines for determining regional, site, and design wind speeds, as well as "
            "wind pressures for protection screens. The Australian/New Zealand Standard AS/NZS 1170.2 provides "
            "methodologies for calculating wind actions on structures, ensuring safety and structural integrity under "
            "various wind conditions. This report documents the wind pressures for Ultimate Limit State (ULS) and "
            "Serviceability Limit State (SLS) at the specified reference height, along with a graph showing the variation "
            "of wind pressure with height up to the reference height. The results are presented for protection screens, "
            "considering factors such as terrain category and wind region. The aerodynamic shape factor is provided by the user, as specified."
        )
    elements.append(Paragraph(intro_text, justified_style))
    elements.append(Spacer(1, 4*mm))

    elements.append(Paragraph("Input Parameters", heading_style))
    has_return_corner_text = "Yes" if inputs['has_return_corner'] else "No"
    input_data_raw = [
        ["Parameter", "Value"],
        ["Location", inputs['location']],
        ["Wind Region", inputs['region']],
        ["Importance Level (ULS)", inputs['importance_level']],
        ["Terrain Category", inputs['terrain_category']],
        ["Reference Height (h, m)", f"{inputs['reference_height']:.2f}"],
        ["Structure Type", inputs['structure_type']],
    ]
    if structure_type == "Free Standing Wall":
        input_data_raw.extend([
            ["Width of the Wall (b, m)", f"{inputs['b']:.2f}"],
            ["Height of the Wall (c, m)", f"{inputs['c']:.2f}"],
            ["Return Corner Extends More Than 1c", has_return_corner_text],
        ])
    elif structure_type == "Protection Screens":
        input_data_raw.append(["Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", f"{inputs['user_C_shp']:.3f}"])
    if inputs['region'] in ["C", "D"]:
        input_data_raw.insert(6, ["Distance from Coast (km)", f"{inputs['distance_from_coast_km']:.2f}"])
    input_data = [[Paragraph(row[0], table_header_style if i == 0 else input_table_cell_style),
                   Paragraph(row[1], table_header_style if i == 0 else input_table_cell_style)] for i, row in enumerate(input_data_raw)]
    input_table = Table(input_data, colWidths=[100*mm, 80*mm])
    input_table.setStyle(input_table_style)
    elements.append(input_table)
    elements.append(Spacer(1, 6*mm))
    elements.append(PageBreak())

    elements.append(Paragraph("Wind Load Results", heading_style))
    limit_states = sorted(results.keys())
    for idx, limit_state in enumerate(limit_states):
        data = results[limit_state]
        elements.append(Paragraph(f"Limit State: {limit_state}", heading_style))
        elements.append(Paragraph(f"Regional Wind Speed (<i>V<sub>R</sub></i>): {data['V_R']:.2f} m/s", normal_style))
        elements.append(Paragraph(f"Site Wind Speed (<i>V<sub>sit,β</sub></i>): {data['V_sit_beta']:.2f} m/s", normal_style))
        elements.append(Paragraph(f"Design Wind Speed (<i>V<sub>des,θ</sub></i>): {data['V_des_theta']:.2f} m/s", normal_style))
        elements.append(Spacer(1, 4*mm))

        if structure_type == "Free Standing Wall":
            thetas = sorted(data['results'].keys())
            for theta in thetas:
                theta_data = data['results'][theta]
                elements.append(Paragraph(f"Wind Direction: θ = {theta}°", normal_style))
                if theta == 0:
                    table_data = [[Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                                   Paragraph("Eccentricity (e, m)", table_header_style),
                                   Paragraph("Wind Pressure (p, kPa)", table_header_style),
                                   Paragraph("Resultant Force (kN)", table_header_style)],
                                  [Paragraph(f"{theta_data['C_shp']:.3f}", table_cell_style),
                                   Paragraph(f"{theta_data['e']:.2f}", table_cell_style),
                                   Paragraph(f"{theta_data['p']:.3f}", table_cell_style),
                                   Paragraph(f"{theta_data['resultant_force']:.2f}", table_cell_style)]]
                    result_table = Table(table_data, colWidths=[45*mm, 35*mm, 35*mm, 35*mm])
                else:
                    table_data = [[Paragraph("Distance from Windward End (m)", table_header_style),
                                   Paragraph("Wind Pressure (p, kPa)", table_header_style)]]
                    distances = theta_data['distances']
                    pressures = theta_data['pressures']
                    step = max(1, len(distances) // 5)
                    for i in range(0, len(distances), step):
                        table_data.append([Paragraph(f"{distances[i]:.2f}", table_cell_style),
                                           Paragraph(f"{pressures[i]:.3f}", table_cell_style)])
                    result_table = Table(table_data, colWidths=[90*mm, 90*mm])
                result_table.setStyle(table_style)
                elements.append(result_table)
                elements.append(Spacer(1, 4*mm))

            elements.append(Paragraph("Pressure Distribution Graph", heading_style))
            graph_filename = data['graph_filename']
            try:
                graph_image = Image(graph_filename, width=140*mm, height=70*mm)
                elements.append(graph_image)
            except Exception as e:
                elements.append(Paragraph(f"[Graph Placeholder - Error: {e}]", normal_style))
            elements.append(Spacer(1, 4*mm))

        else:
            table_data = [[Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                           Paragraph("Eccentricity (e, m)", table_header_style),
                           Paragraph("Wind Pressure (p, kPa)", table_header_style)],
                          [Paragraph(f"{data['C_shp']:.3f}", table_cell_style),
                           Paragraph(f"{data['e']:.2f}", table_cell_style),
                           Paragraph(f"{data['p']:.3f}", table_cell_style)]]
            result_table = Table(table_data, colWidths=[60*mm, 60*mm, 60*mm])
            result_table.setStyle(table_style)
            elements.append(result_table)
            elements.append(Spacer(1, 4*mm))

        if structure_type == "Protection Screens" and idx == len(limit_states) - 1:
            elements.append(Paragraph("Pressure Variation with Height", heading_style))
            graph_filename = results['ULS']['height_pressure_graph']
            try:
                graph_image = Image(graph_filename, width=140*mm, height=70*mm)
                elements.append(graph_image)
            except Exception as e:
                elements.append(Paragraph(f"[Graph Placeholder - Error: {e}]", normal_style))
            elements.append(Spacer(1, 4*mm))

        if idx < len(limit_states) - 1:
            if structure_type != "Protection Screens":
                elements.append(PageBreak())
            else:
                elements.append(Spacer(1, 6*mm))

    return elements

def generate_pdf_report(inputs, results, project_number, project_name):
    try:
        temp_buffer = io.BytesIO()
        temp_doc = SimpleDocTemplate(temp_buffer, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
        elements = build_elements(inputs, results, project_number, project_name)

        def temp_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            page_num = canvas.getPageNumber()
            canvas.drawCentredString(doc.pagesize[0] / 2.0, 8 * mm, f"{PROGRAM} {PROGRAM_VERSION} | tekhne © | Page {page_num}")
            canvas.restoreState()

        temp_doc.build(elements, onFirstPage=temp_footer, onLaterPages=temp_footer)
        total_pages = temp_doc.page

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
        elements = build_elements(inputs, results, project_number, project_name)

        def final_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            page_num = canvas.getPageNumber()
            footer_text = f"{PROGRAM} {PROGRAM_VERSION} | tekhne © | Page {page_num}/{total_pages}"
            canvas.drawCentredString(doc.pagesize[0] / 2.0, 8 * mm, footer_text)
            canvas.restoreState()

        doc.build(elements, onFirstPage=final_footer, onLaterPages=final_footer)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Streamlit UI
def main():
    st.title("Wind Load Calculator (AS/NZS 1170.2:2021)")
    calculator = WindLoadCalculator()
   st.set_page_config(
        page_title="My Page Title",
)

    

    with st.form(key='wind_load_form'):
        # Project details
        project_number = st.text_input("Project Number", value="PRJ-001")
        project_name = st.text_input("Project Name", value="Sample Project")

        # Location
        st.subheader("Location")
        location = st.selectbox("Select Location", calculator.valid_locations, index=calculator.valid_locations.index("Sydney"))

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
            'user_C_shp': user_C_shp
        }

        limit_states = ["ULS", "SLS"]
        results = {}
        for limit_state in limit_states:
            V_R = calculator.determine_V_R(region, limit_state, importance_level, distance_from_coast_km)
            M_d = calculator.determine_M_d(region)
            M_c = calculator.determine_M_c(region)
            M_s = calculator.determine_M_s(region)
            M_t = calculator.determine_M_t(region)
            M_z_cat = calculator.determine_M_z_cat(region, terrain_category, reference_height)
            V_sit_beta = calculator.calculate_site_wind_speed(V_R, M_d, M_c, M_s, M_t, M_z_cat)
            V_des_theta = calculator.calculate_design_wind_speed(V_sit_beta, limit_state)

            if structure_type == "Free Standing Wall":
                thetas = [0, 45, 90]
                theta_results = {}
                for theta in thetas:
                    if theta == 0:
                        C_shp, e = calculator.calculate_aerodynamic_shape_factor(structure_type, None, b, c, h, theta, has_return_corner=has_return_corner)
                        p = calculator.calculate_wind_pressure(V_des_theta, C_shp)
                        resultant_force = p * b * c
                        theta_results[theta] = {'C_shp': C_shp, 'e': e, 'p': p, 'resultant_force': resultant_force}
                    else:
                        distances, pressures = calculator.calculate_pressure_distribution(b, c, h, V_des_theta, theta, has_return_corner=has_return_corner)
                        theta_results[theta] = {'distances': distances, 'pressures': pressures, 'max_pressure': max(pressures)}

                plt.figure(figsize=(8, 4))
                distances, pressures = calculator.calculate_pressure_distribution(b, c, h, V_des_theta, 45, has_return_corner)
                plt.plot(distances, pressures, label="θ = 45°", color="blue")
                distances, pressures = calculator.calculate_pressure_distribution(b, c, h, V_des_theta, 90, has_return_corner)
                plt.plot(distances, pressures, label="θ = 90°", color="green")
                plt.axhline(y=theta_results[0]['p'], color="red", linestyle="--", label="θ = 0° (uniform)")
                plt.xlabel("Distance from Windward Free End (m)")
                plt.ylabel("Wind Pressure (kPa)")
                plt.title(f"Wind Pressure Distribution ({location}, {limit_state})")
                plt.legend()
                plt.grid(True)
                graph_filename = f"pressure_distribution_{limit_state.lower()}.png"
                plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
                plt.close()

                results[limit_state] = {
                    'V_R': V_R, 'V_sit_beta': V_sit_beta, 'V_des_theta': V_des_theta,
                    'results': theta_results, 'graph_filename': graph_filename
                }
            else:
                C_shp, e = calculator.calculate_aerodynamic_shape_factor(structure_type, user_C_shp=user_C_shp)
                p = calculator.calculate_wind_pressure(V_des_theta, C_shp)
                if structure_type == "Protection Screens":
                    heights, V_des_values, pressures = calculator.calculate_pressure_vs_height(
                        region, terrain_category, reference_height, limit_state, importance_level, distance_from_coast_km, C_shp
                    )
                    results[limit_state] = {
                        'V_R': V_R, 'V_sit_beta': V_sit_beta, 'V_des_theta': V_des_theta,
                        'C_shp': C_shp, 'e': e, 'p': p,
                        'heights': heights, 'V_des_values': V_des_values, 'pressures': pressures
                    }
                    if limit_state == "SLS":
                        plt.figure(figsize=(8, 4))
                        plt.plot(results['ULS']['heights'], results['ULS']['pressures'], label="ULS", color="blue")
                        plt.plot(results['SLS']['heights'], results['SLS']['pressures'], label="SLS", color="green")
                        plt.xlabel("Height (m)")
                        plt.ylabel("Wind Pressure (kPa)")
                        plt.title(f"Wind Pressure vs. Height ({location})")
                        plt.legend()
                        plt.grid(True)
                        graph_filename = "height_pressure_graph.png"
                        plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
                        plt.close()
                        results['ULS']['height_pressure_graph'] = graph_filename
                        results['SLS']['height_pressure_graph'] = graph_filename
                else:
                    results[limit_state] = {'V_R': V_R, 'V_sit_beta': V_sit_beta, 'V_des_theta': V_des_theta, 'C_shp': C_shp, 'e': e, 'p': p}

        pdf_data = generate_pdf_report(inputs, results, project_number, project_name)
        if pdf_data:
            st.success("Calculations completed successfully!")
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Wind_Load_Report_{project_number}.pdf",
                mime="application/pdf"
            )
            if structure_type == "Free Standing Wall":
                for limit_state in limit_states:
                    st.image(f"pressure_distribution_{limit_state.lower()}.png", caption=f"Pressure Distribution ({limit_state})")
            elif structure_type == "Protection Screens":
                st.image("height_pressure_graph.png", caption="Pressure vs. Height")
        else:
            st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
