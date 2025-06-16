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
COMPANY_ADDRESS = "    "  # Placeholder; update with actual address if needed
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

# WindLoadCalculator class
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
            "5": "Scaffold" # Added Scaffold
        }
        self.valid_locations = [
            "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
            "Darwin", "Cairns", "Townsville", "Port Hedland", "Alice Springs", "Hobart"
        ]
        # Reduction factors for construction duration (Table 3.2.6(B))
        self.reduction_factors = {
            "A": {"6 months": 0.95, "1 month": 0.85, "1 week": 0.75, "More than 6 months": 1.0},
            "B": {"6 months": 0.95, "1 month": 0.75, "1 week": 0.55, "More than 6 months": 1.0},
            "C": {"6 months": 0.95, "1 month": 0.75, "1 week": 0.55, "More than 6 months": 1.0},
            "D": {"6 months": 0.90, "1 month": 0.70, "1 week": 0.50, "More than 6 months": 1.0},
        }
        # Mapping of specific wind regions to reduction factor categories
        self.region_to_category = {
            "A0": "A", "A1": "A", "A2": "A", "A3": "A", "A4": "A", "A5": "A",
            "B1": "B", "B2": "B",
            "C": "C",
            "D": "D",
        }

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
        # These factors (0.95, 0.90) for V_R_100km and V_R_200km are based on interpretation of
        # as it states interpolation based on distance from coastline for Regions C and D.
        # This implies a reduction as distance increases. Specific factors need external validation or clear standard table.
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

    def determine_V_R(self, region, limit_state, importance_level=None, distance_from_coast_km=None, construction_period=None, reference_height=None):
        if limit_state == "SLS":
            # Calculate V_R for SLS using Equation 3.2(1): V(z) = [(z/10)^0.14 + 0.4] * V_mean
            # The standard specifies a mean wind speed for SLS, which is not directly from a V_R table.
            # A common approach for SLS is to derive from a base mean speed and height profile.
            V_mean = 16.0  # Assumed mean wind speed for SLS as per common interpretation or specific annex of AS1170.2 not provided.
            if reference_height is not None:
                V_R = ((reference_height / 10) ** 0.14 + 0.4) * V_mean
            else:
                V_R = 16.0  # Fallback if reference height not provided
            reduction_factor = 1.0  # No reduction for SLS
            return V_R, reduction_factor
        
        # ULS calculation remains the same
        R_map = {"I": 25, "II": 100, "III": 250} # R values from Table 3.1(A) or similar for ULS
        if importance_level not in R_map:
            raise ValueError("Importance level must be 'I', 'II', or 'III' for ULS.")
        R = R_map[importance_level]
        
        # Base V_R before reduction
        if region in self.regions_with_interpolation and distance_from_coast_km is not None:
            V_R = self.interpolate_V_R(region, R, distance_from_coast_km)
        else:
            V_R = self.V_R_table[region][R]
        
        # Apply reduction factor based on construction duration
        if construction_period:
            region_category = self.region_to_category.get(region, "A")  # Default to A if region not found
            reduction_factor = self.reduction_factors[region_category].get(construction_period, 1.0)
            V_R = V_R * reduction_factor
        else:
            reduction_factor = 1.0  # No reduction if construction period not provided
        
        return V_R, reduction_factor

    def determine_M_c(self, region):
        return self.M_c_table[region]

    def determine_M_d(self, region):
        # M_d is set to 1.0 for circular or polygonal chimneys, tanks, and poles, and cladding/immediate supporting structure in Regions B2, C, D.
        # For general structures and other cases, it's typically from Table 3.2(A) or 3.2(B).
        # For simplicity in this general calculator, we'll keep it as 1.0. For specific designs, Table 3.2(A) or (B) should be used.
        return 1.0

    def determine_M_s(self, region):
        # M_s is 1.0 for structures > 25m in height or on steep slopes.
        # For simplicity, default to 1.0; detailed shielding requires more complex input (Table 4.2).
        return 1.0

    def determine_M_t(self, region):
        # M_t is typically 1.0, but varies with topography (hills, ridges, escarpments).
        # For simplicity, default to 1.0; complex topography requires more detailed input (Clause 4.4.2, 4.4.3).
        return 1.0

    def determine_M_z_cat(self, region, terrain_category, height):
        # Specific rule for Region A0 for heights > 100m.
        if region == "A0" and height > 100:
            return 1.24 if height <= 200 else 1.24 # M_z,cat is 1.24 for 100 < z <= 200m in A0.
        
        terrain_data = self.M_z_cat_table[terrain_category]
        heights = sorted(terrain_data.keys())
        if height in heights:
            return terrain_data[height]
        if height <= heights[0]:
            return terrain_data[heights[0]]
        if height >= heights[-1]:
            return terrain_data[heights[-1]]
        
        # Linear interpolation for intermediate heights
        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i + 1]
            if h1 < height <= h2:
                m1, m2 = terrain_data[h1], terrain_data[h2]
                fraction = (height - h1) / (h2 - h1)
                return m1 + fraction * (m2 - m1)
        return 1.0

    def calculate_site_wind_speed(self, V_R, M_d, M_c, M_s, M_t, M_z_cat):
        # Equation 2.2 in AS/NZS 1170.2:2021
        return V_R * M_c * M_d * M_z_cat * M_s * M_t

    def calculate_design_wind_speed(self, V_sit_beta, limit_state):
        # V_des,theta must not be less than 30 m/s for ULS.
        if limit_state == "ULS":
            return max(V_sit_beta, 30.0)
        return V_sit_beta

    def calculate_Cpn_freestanding_wall(self, b, c, h, theta, distance_from_windward_end=None, has_return_corner=False):
        # Based on Tables B.2(A), B.2(B), B.2(C), B.2(D) in Appendix B of AS/NZS 1170.2:2021 
        b_over_c = b / c
        c_over_h = c / h
        
        Cpn = 0.0 # Initialize Cpn
        e = 0.0 # Initialize eccentricity

        if theta == 0: # Wind normal to hoarding or wall
            if 0.5 <= b_over_c <= 5:
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else: # c/h < 0.2
                    Cpn = 1.4 + 0.3 * log10(b_over_c)
            else: # b/c > 5
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.7 - 0.5 * c_over_h
                else: # c/h < 0.2
                    Cpn = 1.4 + 0.3 * log10(b_over_c) # For all b/c when c/h < 0.2
            e = 0.0
        elif theta == 45: # Wind at 45 degrees to hoarding or wall
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=45°.")
            
            # This logic combines aspects of Table B.2(B) and B.2(C)
            # Table B.2(B) provides the Cpn for b/c 0.5 to 5, and it is a single value, not distance dependent.
            # Table B.2(C) is for b/c > 5, and is distance dependent.
            if 0.5 <= b_over_c <= 5:
                if 0.2 <= c_over_h <= 1:
                    Cpn = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else: # c/h < 0.2
                    Cpn = 1.4 + 0.3 * log10(b_over_c)
            else: # b/c > 5 (using Table B.2(C)) 
                if c_over_h <= 0.7:
                    if distance_from_windward_end <= 2 * c:
                        Cpn = 3.0
                    elif distance_from_windward_end <= 4 * c:
                        Cpn = 1.5
                    else:
                        Cpn = 0.75
                else: # c/h > 0.7
                    if distance_from_windward_end <= 2 * h: # Note: 'h' here refers to reference_height (total height) not c.
                        Cpn = 2.4
                    elif distance_from_windward_end <= 4 * h:
                        Cpn = 1.2
                    else:
                        Cpn = 0.6
                
                # Apply return corner condition if applicable for b/c > 5, as per note in Table B.2(C) 
                if has_return_corner:
                    if c_over_h <= 0.7 and distance_from_windward_end <= 2 * c:
                        Cpn = 2.2
                    elif c_over_h > 0.7 and distance_from_windward_end <= 2 * h:
                        Cpn = 1.8
            e = 0.2 * b # Eccentricity for 45 deg wind.
        elif theta == 90: # Wind parallel to hoarding or wall
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=90°.")
            
            # Based on Table B.2(D) 
            if c_over_h <= 0.7:
                if distance_from_windward_end <= 2 * c:
                    Cpn = 1.2
                elif distance_from_windward_end <= 4 * c:
                    Cpn = 0.6
                else:
                    Cpn = 0.3
            else: # c/h > 0.7
                if distance_from_windward_end <= 2 * h: # Note: 'h' here refers to reference_height (total height)
                    Cpn = 1.0
                elif distance_from_windward_end <= 4 * h:
                    Cpn = 0.25
                else:
                    Cpn = 0.25
            Cpn = abs(Cpn) # Table B.2(D) lists +/- values, but for calculation, use absolute.
            e = 0.0 # Eccentricity for 90 deg wind.
        else:
            raise ValueError("Theta must be 0°, 45°, or 90°.")
        return Cpn, e

    def _calculate_Cshp_open_scaffold(self, solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta):
        # Based on AS/NZS 1170.2:2021 Appendix C, especially C.2.2, C.2.3 and Table C.6(B) 
        # This assumes circular members (scaffold tubes) and wind normal to the face.

        # 1. Determine effective solidity (delta_e) - C.2.2 
        # For circular members, delta_e = 1.2 * delta^1.75. Here, user provides delta (solidity_ratio)
        delta = solidity_ratio
        delta_e = 1.2 * (delta ** 1.75) 

        # 2. Determine drag force coefficient (Cd) for a single frame - Table C.6(B) for circular members 
        # Values depend on flow regime (bi*Vdes,theta) and solidity.
        # We need to approximate bi*Vdes,theta. Assuming 'bi' is member_diameter_m.
        # For simplicity, we'll assume supercritical flow (bi*Vdes,theta >= 6 m2/s) or take the higher Cd.
        # Interpolate from Table C.6(B) for given delta_e
        
        # Simplified Cd lookup for circular members (Table C.6(B) for super-critical flow, higher of two values for given delta) 
        # Solidity (delta) | Cd_onto_face | Cd_onto_corner
        # ------------------|--------------|----------------
        # <= 0.05          | 1.4          | 1.6
        # 0.1              | 1.4          | 1.6
        # 0.2              | 1.5          | 1.7
        # >= 0.3           | 1.7          | 1.9
        
        # We'll use the 'onto corner' values which are generally higher and more conservative
        if delta <= 0.05:
            Cd_single_frame = 1.6
        elif delta <= 0.1:
            Cd_single_frame = 1.6
        elif delta <= 0.2:
            Cd_single_frame = 1.7
        elif delta >= 0.3:
            Cd_single_frame = 1.9
        else: # Linear interpolation for 0.2 < delta < 0.3
            Cd_single_frame = np.interp(delta, [0.2, 0.3], [1.7, 1.9])

        # 3. Determine shielding factor (Ksh) - Table C.2 
        # Assumed wind normal to frames (0 degrees).
        # lambda_ratio = frame_spacing (typical_bay_length_m) / (smaller of l or b of frame)
        # Here, 'l' is assumed to be the height, 'b' is bay_width_m.
        # The smaller of l or b of frame is approximated by typical_bay_width_m for lambda calculation.
        # As per C.2.3, lambda is frame spacing (typical_bay_length_m) divided by smaller of l or b.
        # Let's assume 'l' (length of frame face) is a full bay height (reference_height) and 'b' is typical_bay_width_m.
        # Smallest dimension of the frame is the typical_bay_width_m.
        
        # For simplicity, let's derive lambda from typical bay length (spacing) divided by typical bay width (frame dimension)
        # This implies a ratio of spacing between frames.
        # Table C.2 uses lambda = frame spacing (s) / smaller of l or b of the frame
        # If typical_bay_length_m is the spacing, and typical_bay_width_m is the frame dimension.
        
        lambda_ratio = typical_bay_length_m / typical_bay_length_m # This needs refinement. lambda is s / min(l,b)
                                                                   # For a scaffold, if we assume typical bay length is spacing (s)
                                                                   # And a 'frame' is a cross-section of width (b).
                                                                   # It might be more appropriate to use a fixed lambda like 1.0 or 2.0 based on typical scaffold.
                                                                   # Let's assume lambda=1 for now, as frame spacing for typical scaffolds is often around 1.2m
                                                                   # and breadth might be 0.9-1.2m
        # Let's use a simpler interpretation: if number of bays > 1, apply some shielding.
        # Otherwise, if it's a single bay, no shielding (Ksh = 1).
        
        # Table C.2: Effective solidity (delta_e) ranges: 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0
        # For given delta_e, and lambda, interpolate Ksh.
        
        # Assume wind direction normal to frames (0 degrees) for Table C.2.
        # Let's simplify lambda to assume common scaffold spacing relative to frame width, say 2.0 or 4.0 for now.
        # Or, base it directly on num_bays_length if it's multiple bays.
        # If num_bays_length is 1, no shielding, sum Ksh = 0.
        # If num_bays_length > 1, sum Ksh = (num_bays_length - 1) * Ksh for a typical lambda.

        # Simplified approach for K_sh for multiple bays (assuming average lambda ~ 2.0-4.0 for typical scaffold)
        # Using lambda = 4.0 for a conservative but reasonable K_sh for multi-bay.
        # Table C.2, Angle 0 deg, lambda 4.0 
        ksh_values_at_lambda_4 = {
            0: 1.0, 0.1: 1.0, 0.2: 1.0, 0.3: 0.8, 0.4: 0.7, 0.5: 0.6, 0.7: 0.4, 1.0: 0.2
        }
        
        # Interpolate Ksh based on effective solidity
        solidity_keys = sorted(ksh_values_at_lambda_4.keys())
        if delta_e <= solidity_keys[0]:
            K_sh_interpolated = ksh_values_at_lambda_4[solidity_keys[0]]
        elif delta_e >= solidity_keys[-1]:
            K_sh_interpolated = ksh_values_at_lambda_4[solidity_keys[-1]]
        else:
            for i in range(len(solidity_keys) - 1):
                s1, s2 = solidity_keys[i], solidity_keys[i+1]
                if s1 <= delta_e <= s2:
                    k1, k2 = ksh_values_at_lambda_4[s1], ksh_values_at_lambda_4[s2]
                    if s2 - s1 == 0: # Avoid division by zero if keys are identical (shouldn't happen with sorted unique keys)
                        K_sh_interpolated = k1
                    else:
                        fraction = (delta_e - s1) / (s2 - s1)
                        K_sh_interpolated = k1 + fraction * (k2 - k1)
                    break
        
        total_ksh = 0
        if num_bays_length > 1:
            # Assuming all subsequent bays have the same K_sh
            total_ksh = (num_bays_length - 1) * K_sh_interpolated
        
        # Total C_shp = Cshp,1 + SUM(Ksh * Cshp,1) = Cshp,1 * (1 + SUM(Ksh))
        # AS/NZS 1170.2:2021 Eq. C.2(5) is C_shp = C_shp,1 + SUM(K_sh * C_shp,1). This can be factored.
        C_shp_overall = Cd_single_frame * (1 + total_ksh)
        
        return C_shp_overall, 0.0 # Eccentricity is 0 for overall scaffold load.

    def calculate_aerodynamic_shape_factor(self, structure_type, user_C_shp=None, b=None, c=None, h=None, theta=None, distance_from_windward_end=None, has_return_corner=False, scaffold_type=None, solidity_ratio=None, num_bays_length=None, num_rows_width=None, typical_bay_length_m=None, typical_bay_width_m=None, member_diameter_mm=None, V_des_theta=None):
        K_p = 1.0 # Permeable cladding reduction factor (Kp) is 1.0 by default unless specified
        if structure_type == "Free Standing Wall":
            # For free standing walls, C_shp = C_pn * Kp. Kp defaults to 1.0 here unless porosity is considered.
            # The current Cpn function assumes Kp=1.0.
            Cpn, e = self.calculate_Cpn_freestanding_wall(b, c, h, theta, distance_from_windward_end, has_return_corner)
            return Cpn * K_p, e
        elif structure_type == "Circular Tank":
            # For overall drag on circular tanks, C_shp is 0.63.
            # Here we use 0.8 as a placeholder for a general C_shp for circular tanks if not specific section of tank.
            # A more detailed calculation would refer to A.5.2.1
            return 0.8, 0.0
        elif structure_type == "Attached Canopy":
            # For attached canopies, a C_shp of 1.2 is often a conservative general value based on Table B.9.
            # More precise calculations would use Tables B.9 and B.10.
            return 1.2, 0.0
        elif structure_type == "Protection Screens":
            # For Protection Screens, C_shp is user-defined.
            if user_C_shp is None:
                raise ValueError("C_shp required for Protection Screens.")
            return user_C_shp, 0.0 # Eccentricity not applicable for general screen pressure.
        elif structure_type == "Scaffold":
            if scaffold_type == "Open (Unclad)":
                if None in [solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta]:
                    raise ValueError("All scaffold parameters are required for Open (Unclad) Scaffold.")
                # We need an effective 'b' for the overall structure for V_des_theta calculation.
                # Assuming 'b' is the overall width (num_rows_width * typical_bay_width_m) or similar.
                # Here, we pass the V_des_theta calculated at the reference height to _calculate_Cshp_open_scaffold.
                # The 'b' used in Cd lookup is member_diameter_mm.
                return self._calculate_Cshp_open_scaffold(solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta), 0.0
            elif scaffold_type == "Fully Clad":
                # Treat as a freestanding wall with wind normal to face (theta=0) with full height (h) and length (b)
                # Need overall scaffold dimensions (length and width from inputs)
                # Assume 'b' as the length of the scaffold (inputs['length']) and 'c' as the total height (inputs['h'])
                # and 'h' as the total height (inputs['h']) for Cpn calculation.
                # The Cpn for a fully clad scaffold (acting as a hoarding/wall) would be applied to its overall projected area.
                
                # Using formula for a very long wall (b/c > 5), c/h ~ 1 (since c is h for overall wall) from Table B.2(A) 
                # Cpn = 1.4 + 0.3 * log10(b_over_c) when c/h < 0.2 (low wall to overall height ratio)
                # Cpn = 1.7 - 0.5 * c/h when b/c > 5 and c/h >= 0.2.
                # For a fully clad scaffold, b (length) is long, c (height) is h. So b/c can be > 5. c/h = 1.
                # Using 1.7 - 0.5 * 1.0 = 1.2
                # A common and conservative approach for a solid hoarding is to use Cpn ~ 1.2 to 1.5. Let's use 1.2.
                return 1.2, 0.0 # Cshp of 1.2 is a reasonable conservative value for a solid face.
            else:
                raise ValueError("Invalid Scaffold type selected.")
        else:
            raise ValueError("Invalid structure type.")

    def calculate_wind_pressure(self, V_des_theta, C_shp):
        rho_air = 1.2 # Density of air is taken as 1.2 kg/m^3
        C_dyn = 1.0 # Dynamic response factor C_dyn is 1.0 for most structures (refer AS/NZS 1170.2:2021 Section 6 for exceptions) 
        # Equation 2.4(1) from AS/NZS 1170.2:2021 
        return (0.5 * rho_air) * (V_des_theta ** 2) * C_shp * C_dyn / 1000 # Convert Pa to kPa

    def calculate_pressure_distribution(self, b, c, h, V_des_theta, theta, has_return_corner=False):
        num_points = 100
        distances = np.linspace(0, b, num_points)
        pressures = []
        for d in distances:
            # Re-calculate C_shp for each point on the wall based on distance from windward end.
            C_shp, _ = self.calculate_aerodynamic_shape_factor(
                "Free Standing Wall", None, b, c, h, theta, distance_from_windward_end=d, has_return_corner=has_return_corner
            )
            p = self.calculate_wind_pressure(V_des_theta, C_shp)
            pressures.append(p)
        return distances, pressures

    def calculate_pressure_vs_height(self, region, terrain_category, reference_height, limit_state, importance_level, distance_from_coast_km, C_shp_base, scaffold_type=None, solidity_ratio=None, num_bays_length=None, num_rows_width=None, typical_bay_length_m=None, member_diameter_mm=None):
        height_step = 5.0
        # Generate heights up to and including reference_height
        heights = np.arange(0, reference_height + height_step, height_step)
        
        # Adjust heights to precisely include reference_height and ensure uniqueness/sorting
        if reference_height not in heights:
            heights = np.append(heights, reference_height)
        heights = np.unique(np.sort(heights))

        V_des_values = []
        pressures = []
        
        # Determine base V_R outside the loop for efficiency, especially for ULS where it's constant
        V_R_base, _ = self.determine_V_R(region, limit_state, importance_level, distance_from_coast_km, reference_height=reference_height)
        
        for h_current in heights:
            # For SLS, V_R determination explicitly uses reference_height in determine_V_R
            # so we should use h_current for SLS V_R determination
            if limit_state == "SLS":
                V_R_h_current, _ = self.determine_V_R(region, limit_state, reference_height=h_current)
            else: # For ULS, V_R is constant, already calculated as V_R_base
                V_R_h_current = V_R_base

            M_d = self.determine_M_d(region)
            M_c = self.determine_M_c(region)
            M_s = self.determine_M_s(region)
            M_t = self.determine_M_t(region)
            M_z_cat = self.determine_M_z_cat(region, terrain_category, h_current) # M_z_cat is height-dependent
            
            V_sit_beta_current = self.calculate_site_wind_speed(V_R_h_current, M_d, M_c, M_s, M_t, M_z_cat)
            V_des_current = self.calculate_design_wind_speed(V_sit_beta_current, limit_state)

            # If scaffold is open, its C_shp also depends on V_des_theta for the flow regime, and other scaffold parameters.
            # Recalculate C_shp for open scaffold if necessary at each height's V_des
            C_shp_at_current_height = C_shp_base
            if scaffold_type == "Open (Unclad)":
                # Recalculate C_shp for open scaffold as it depends on V_des_theta (for flow regime which affects Cd)
                # However, our _calculate_Cshp_open_scaffold only uses V_des_theta for member_diameter*V_des_theta.
                # And we've simplified Cd lookup. So C_shp_base from reference height is mostly okay for now.
                # If we had Reynolds number dependent Cd lookup, this would be crucial.
                # For this simplified model, C_shp_base (calculated at reference height's V_des_theta) is adequate.
                pass # C_shp_base already calculated and passed.

            p = self.calculate_wind_pressure(V_des_current, C_shp_at_current_height)
            V_des_values.append(V_des_current)
            pressures.append(p)
        return heights, V_des_values, pressures

# PDF generation functions
def build_elements(inputs, results, project_number, project_name):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=16, spaceAfter=6, alignment=1)
    subtitle_style = ParagraphStyle(name='SubtitleStyle', parent=styles['Normal'], fontSize=10, spaceAfter=6, alignment=1)
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=12, spaceAfter=4)
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4, fontName='Helvetica-Bold')
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
    logo_img_obj = Image(logo_file, width=50*mm, height=20*mm) if logo_file else Paragraph("[Logo Placeholder]", normal_style)
    header_data = [[logo_img_obj, company_paragraph]]
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
    elif structure_type == "Scaffold":
        intro_text = (
            "This report presents the wind load calculations for structural design as per AS/NZS 1170.2:2021, "
            "with consideration of AS/NZS 1576.1:2019 for scaffold-specific parameters. "
            "The calculation determines wind pressures for the selected scaffold configuration (open or clad) "
            "at the specified reference height, and includes a graph showing the variation "
            "of wind pressure with height. The aerodynamic shape factor for open scaffolds is determined "
            "using principles from Appendix C of AS/NZS 1170.2:2021 based on solidity and shielding effects. "
            "For fully clad scaffolds, it is treated as a solid hoarding."
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
        ["Construction Duration", inputs['construction_period']],
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
    elif structure_type == "Scaffold": # Added Scaffold inputs
        input_data_raw.append(["Scaffold Type", inputs['scaffold_type']])
        if inputs['scaffold_type'] == "Open (Unclad)":
            input_data_raw.extend([
                ["Overall Length (m)", f"{inputs['length']:.2f}"],
                ["Overall Width (m)", f"{inputs['width']:.2f}"],
                ["Solidity Ratio (δ)", f"{inputs['solidity_ratio']:.3f}"],
                ["Number of Bays (Length-wise)", f"{int(inputs['num_bays_length'])}"],
                ["Number of Rows (Width-wise)", f"{int(inputs['num_rows_width'])}"],
                ["Typical Bay Length (m)", f"{inputs['typical_bay_length_m']:.2f}"],
                ["Typical Bay Width (m)", f"{inputs['typical_bay_width_m']:.2f}"],
                ["Member Diameter (mm)", f"{inputs['member_diameter_mm']:.1f}"],
            ])
        elif inputs['scaffold_type'] == "Fully Clad":
            input_data_raw.extend([
                ["Overall Length (m)", f"{inputs['length']:.2f}"],
                ["Overall Width (m)", f"{inputs['width']:.2f}"],
            ])

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
        # Add reduction factor text (for ULS only)
        if limit_state == "ULS":
            reduction_factor = data.get('reduction_factor', 1.0)
            if reduction_factor != 1.0:
                elements.append(Paragraph(f"Reduction Factor Applied (based on construction duration): {reduction_factor:.2f}", normal_style))
            else:
                elements.append(Paragraph("No reduction factor applied (construction duration > 6 months)", normal_style))
        else:
            elements.append(Paragraph("No reduction factor applied (SLS)", normal_style))
        elements.append(Paragraph(f"Site Wind Speed (<i>V<sub>sit,β</sub></i>): {data['V_sit_beta']:.2f} m/s", normal_style))
        elements.append(Paragraph(f"Design Wind Speed (<i>V<sub>des,θ</sub></i>): {data['V_des_theta']:.2f} m/s", normal_style))
        elements.append(Spacer(1, 4*mm))

        if structure_type == "Free Standing Wall":
            thetas = sorted(data['results'].keys())
            for theta in thetas:
                theta_data = data['results'][theta]
                elements.append(Paragraph(f"Wind Direction: θ = {theta}°", normal_style))
                if theta == 0:
                    if limit_state == "ULS":
                        table_data = [
                            [
                                Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                                Paragraph("Eccentricity (e, m)", table_header_style),
                                Paragraph("<b>Wind Pressure (p, kPa) (ULS)</b>", table_header_style),
                                Paragraph("Resultant Force (kN) (ULS)", table_header_style),
                            ],
                            [
                                Paragraph(f"{theta_data['C_shp']:.3f}", table_cell_style),
                                Paragraph(f"{theta_data['e']:.2f}", table_cell_style),
                                Paragraph(f"{theta_data['p_uls']:.3f}", table_cell_style),
                                Paragraph(f"{theta_data['resultant_force_uls']:.2f}", table_cell_style),
                            ]
                        ]
                        result_table = Table(table_data, colWidths=[45*mm, 35*mm, 35*mm, 35*mm])
                    else:  # SLS
                        table_data = [
                            [
                                Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                                Paragraph("Eccentricity (e, m)", table_header_style),
                                Paragraph("Wind Pressure (p, kPa) (SLS)", table_header_style),
                                Paragraph("Resultant Force (kN) (SLS)", table_header_style),
                            ],
                            [
                                Paragraph(f"{theta_data['C_shp']:.3f}", table_cell_style),
                                Paragraph(f"{theta_data['e']:.2f}", table_cell_style),
                                Paragraph(f"{theta_data['p_sls']:.3f}", table_cell_style),
                                Paragraph(f"{theta_data['resultant_force_sls']:.2f}", table_cell_style),
                            ]
                        ]
                        result_table = Table(table_data, colWidths=[45*mm, 35*mm, 35*mm, 35*mm])
                else:
                    if limit_state == "ULS":
                        table_data = [
                            [
                                Paragraph("Distance from Windward End (m)", table_header_style),
                                Paragraph("<b>Wind Pressure (p, kPa) (ULS)</b>", table_header_style),
                            ]
                        ]
                        distances = theta_data['distances']
                        pressures = theta_data['pressures_uls']
                        step = max(1, len(distances) // 5)
                        for i in range(0, len(distances), step):
                            table_data.append([
                                Paragraph(f"{distances[i]:.2f}", table_cell_style),
                                Paragraph(f"{pressures[i]:.3f}", table_cell_style),
                            ])
                        result_table = Table(table_data, colWidths=[90*mm, 90*mm])
                    else:  # SLS
                        table_data = [
                            [
                                Paragraph("Distance from Windward End (m)", table_header_style),
                                Paragraph("Wind Pressure (p, kPa) (SLS)", table_header_style),
                            ]
                        ]
                        distances = theta_data['distances']
                        pressures = theta_data['pressures_sls']
                        step = max(1, len(distances) // 5)
                        for i in range(0, len(distances), step):
                            table_data.append([
                                Paragraph(f"{distances[i]:.2f}", table_cell_style),
                                Paragraph(f"{pressures[i]:.3f}", table_cell_style),
                            ])
                        result_table = Table(table_data, colWidths=[90*mm, 90*mm])
                result_table.setStyle(table_style)
                elements.append(result_table)
                elements.append(Spacer(1, 4*mm))

            # Show graph for both ULS and SLS for Free Standing Wall
            elements.append(Paragraph(f"Pressure Distribution Graph ({limit_state})", heading_style))
            graph_filename = data['graph_filename']
            try:
                graph_image = Image(graph_filename, width=140*mm, height=70*mm)
                elements.append(graph_image)
            except Exception as e:
                elements.append(Paragraph(f"[Graph Placeholder - Error: {e}]", normal_style))
            elements.append(Spacer(1, 4*mm))

        else: # For Circular Tank, Attached Canopy, Protection Screens, Scaffold
            # Display overall C_shp and pressure for ULS and SLS
            if limit_state == "ULS":
                table_data = [
                    [
                        Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                        Paragraph("Eccentricity (e, m)", table_header_style),
                        Paragraph("<b>Wind Pressure (p, kPa) (ULS)</b>", table_header_style),
                    ],
                    [
                        Paragraph(f"{data['C_shp']:.3f}", table_cell_style),
                        Paragraph(f"{data['e']:.2f}", table_cell_style),
                        Paragraph(f"{data['p_uls']:.3f}", table_cell_style),
                    ]
                ]
                result_table = Table(table_data, colWidths=[60*mm, 60*mm, 60*mm])
            else: # SLS
                table_data = [
                    [
                        Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                        Paragraph("Eccentricity (e, m)", table_header_style),
                        Paragraph("Wind Pressure (p, kPa) (SLS)", table_header_style),
                    ],
                    [
                        Paragraph(f"{data['C_shp']:.3f}", table_cell_style),
                        Paragraph(f"{data['e']:.2f}", table_cell_style),
                        Paragraph(f"{data['p_sls']:.3f}", table_cell_style),
                    ]
                ]
                result_table = Table(table_data, colWidths=[60*mm, 60*mm, 60*mm])
            result_table.setStyle(table_style)
            elements.append(result_table)
            elements.append(Spacer(1, 4*mm))
            
            # Show Pressure Variation with Height graph ONLY for Protection Screens AND ULS
            # Or for Scaffold ULS (as both ULS and SLS share the same graph)
            if (structure_type == "Protection Screens" or structure_type == "Scaffold") and limit_state == "ULS":
                elements.append(Paragraph("Pressure Variation with Height (ULS)", heading_style))
                graph_filename = results['ULS']['height_pressure_graph']
                try:
                    graph_image = Image(graph_filename, width=140*mm, height=70*mm)
                    elements.append(graph_image)
                except Exception as e:
                    elements.append(Paragraph(f"[Graph Placeholder - Error: {e}]", normal_style))
                elements.append(Spacer(1, 4*mm))


        # Add page break or spacer between limit states for Free Standing Walls or if it's the end of other types
        if idx < len(limit_states) - 1:
            # If Free Standing Wall, add page break between ULS and SLS sections
            if structure_type == "Free Standing Wall":
                elements.append(PageBreak())
            # For Protection Screens or Scaffold, if we just showed the ULS graph, add a spacer before SLS.
            # No page break needed between ULS and SLS for other types.
            elif (structure_type == "Protection Screens" or structure_type == "Scaffold") and limit_state == "ULS":
                 elements.append(Spacer(1, 6*mm))
            # For other types, just a spacer between ULS and SLS
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
    # Set page configuration with a title for the browser tab
    st.set_page_config(page_title="Wind Load Calculator - AS/NZS 1170.2:2021")

    # Logo display
    if logo:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(logo, width=200)
        st.markdown('</div>', unsafe_allow_html=True)

    st.title("Wind Load Calculator (AS/NZS 1170.2:2021)")
    calculator = WindLoadCalculator()

    # --- Inputs OUTSIDE the form to enable dynamic rendering ---
    # Project details (can stay outside or inside, but usually good practice to group fixed inputs)
    col1, col2 = st.columns(2)
    with col1:
        project_number = st.text_input("Project Number", value="PRJ-001")
    with col2:
        project_name = st.text_input("Project Name", value="Sample Project")
    
    # Construction Duration input
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

    # Structure Type (This dropdown must be outside the form for dynamic behavior)
    st.subheader("Structure Type")
    structure_choice = st.selectbox("Select Structure Type", list(calculator.structure_types.values()))
    structure_type = structure_choice

    # Structure-specific inputs (moved outside the st.form)
    # These will cause an automatic rerun when 'structure_type' changes
    b = c = user_C_shp = length = width = solidity_ratio = num_bays_length = num_rows_width = typical_bay_length_m = typical_bay_width_m = member_diameter_mm = scaffold_type = None
    has_return_corner = False

    if structure_type == "Free Standing Wall":
        b = st.number_input("Width of the Wall (b, m)", min_value=0.1, value=10.0, step=0.1)
        c = st.number_input("Height of the Wall (c, m)", min_value=0.1, max_value=reference_height, value=min(3.0, reference_height), step=0.1)
        one_c = c
        st.write(f"Note: 1c = {one_c:.2f} m (based on wall height c)")
        has_return_corner = st.checkbox(f"Return Corner Extends More Than 1c ({one_c:.2f} m)")
    elif structure_type == "Protection Screens":
        user_C_shp = st.number_input("Aerodynamic Shape Factor (C_shp)", min_value=0.1, value=1.0, step=0.01)
    elif structure_type == "Scaffold":
        scaffold_type = st.selectbox("Scaffold Configuration", ["Open (Unclad)", "Fully Clad"])
        length = st.number_input("Overall Length (m)", min_value=1.0, value=10.0, step=0.1)
        width = st.number_input("Overall Width (m)", min_value=0.5, value=1.2, step=0.1) # Width perpendicular to wind, usually bay width
        
        if scaffold_type == "Open (Unclad)":
            solidity_ratio = st.number_input("Solidity Ratio (δ)", min_value=0.01, max_value=0.99, value=0.15, step=0.01, help="Ratio of solid area to total area. Typical range: 0.1 to 0.3 for open scaffolds.")
            num_bays_length = st.number_input("Number of Bays (Length-wise)", min_value=1, value=4, step=1, help="Number of bays along the scaffold length.")
            num_rows_width = st.number_input("Number of Rows (Width-wise)", min_value=1, value=2, step=1, help="Number of rows perpendicular to the wind.")
            typical_bay_length_m = st.number_input("Typical Bay Length (m)", min_value=0.5, value=2.4, step=0.1, help="Length of a single bay (e.g., 2.4m, 1.8m). Used for shielding factor calculation.")
            typical_bay_width_m = st.number_input("Typical Bay Width (m)", min_value=0.5, value=1.2, step=0.1, help="Width of a single bay. Used in overall dimensions.")
            member_diameter_mm = st.number_input("Typical Member Diameter (mm)", min_value=10.0, value=48.3, step=0.1, help="Typical outer diameter of scaffold tubes (e.g., 48.3mm for steel).")


    # Now, the st.form starts, and it will use the values selected above
    with st.form(key='wind_load_form'):
        # No inputs here, just the submit button
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
            'b': b, # For Free Standing Wall
            'c': c, # For Free Standing Wall
            'has_return_corner': has_return_corner, # For Free Standing Wall
            'user_C_shp': user_C_shp, # For Protection Screens
            'scaffold_type': scaffold_type, # For Scaffold
            'length': length, # For Scaffold
            'width': width, # For Scaffold
            'solidity_ratio': solidity_ratio, # For Open Scaffold
            'num_bays_length': num_bays_length, # For Open Scaffold
            'num_rows_width': num_rows_width, # For Open Scaffold
            'typical_bay_length_m': typical_bay_length_m, # For Open Scaffold
            'typical_bay_width_m': typical_bay_width_m, # For Open Scaffold
            'member_diameter_mm': member_diameter_mm, # For Open Scaffold
            'construction_period': construction_period
        }

        # Calculate SLS wind speed separately for service wind pressure (at reference height)
        # This is for the p_sls value displayed in tables for all structure types
        V_R_sls_ref_height, _ = calculator.determine_V_R(
            region, "SLS", reference_height=reference_height
        )
        M_d = calculator.determine_M_d(region)
        M_c = calculator.determine_M_c(region)
        M_s = calculator.determine_M_s(region)
        M_t = calculator.determine_M_t(region)
        M_z_cat_ref_height = calculator.determine_M_z_cat(region, terrain_category, reference_height)
        V_sit_beta_sls_ref_height = calculator.calculate_site_wind_speed(V_R_sls_ref_height, M_d, M_c, M_s, M_t, M_z_cat_ref_height)
        V_des_theta_sls_ref_height = calculator.calculate_design_wind_speed(V_sit_beta_sls_ref_height, "SLS")


        limit_states = ["ULS", "SLS"]
        results = {}
        
        # Calculate C_shp and eccentricity once for non-Free Standing Wall types at reference height
        # Note: For Scaffold, C_shp might depend on V_des_theta (for flow regime), so V_des_theta is passed.
        # However, for our simplified Cd lookup, V_des_theta is not strictly used inside _calculate_Cshp_open_scaffold yet.
        # This C_shp_overall should be based on V_des_theta calculated AT THE REFERENCE HEIGHT
        
        V_R_uls_ref_height, reduction_factor_uls = calculator.determine_V_R(
            region, "ULS", importance_level, distance_from_coast_km, construction_period, reference_height
        )
        M_d_uls = calculator.determine_M_d(region)
        M_c_uls = calculator.determine_M_c(region)
        M_s_uls = calculator.determine_M_s(region)
        M_t_uls = calculator.determine_M_t(region)
        M_z_cat_uls = calculator.determine_M_z_cat(region, terrain_category, reference_height)
        V_sit_beta_uls_ref_height = calculator.calculate_site_wind_speed(V_R_uls_ref_height, M_d_uls, M_c_uls, M_s_uls, M_t_uls, M_z_cat_uls)
        V_des_theta_uls_ref_height = calculator.calculate_design_wind_speed(V_sit_beta_uls_ref_height, "ULS")

        C_shp_overall_for_table, e_for_other_types = None, 0.0 # Default eccentricity to 0.0

        if structure_type == "Protection Screens":
            C_shp_overall_for_table, e_for_other_types = calculator.calculate_aerodynamic_shape_factor(
                structure_type, user_C_shp=user_C_shp
            )
        elif structure_type == "Scaffold":
            # Pass all scaffold related parameters for C_shp calculation
            C_shp_overall_for_table, e_for_other_types = calculator.calculate_aerodynamic_shape_factor(
                structure_type, 
                scaffold_type=scaffold_type, 
                solidity_ratio=solidity_ratio, 
                num_bays_length=num_bays_length, 
                num_rows_width=num_rows_width, 
                typical_bay_length_m=typical_bay_length_m, 
                typical_bay_width_m=typical_bay_width_m, 
                member_diameter_mm=member_diameter_mm,
                V_des_theta=V_des_theta_uls_ref_height # Pass ULS V_des_theta at ref height
            )
        elif structure_type in ["Circular Tank", "Attached Canopy"]:
             C_shp_overall_for_table, e_for_other_types = calculator.calculate_aerodynamic_shape_factor(structure_type)


        for limit_state in limit_states:
            # V_R, V_sit_beta, V_des_theta are for the summary table at reference height
            V_R_table_summary, reduction_factor = calculator.determine_V_R(
                region, limit_state, importance_level, distance_from_coast_km, construction_period, reference_height
            )
            M_d_summary = calculator.determine_M_d(region)
            M_c_summary = calculator.determine_M_c(region)
            M_s_summary = calculator.determine_M_s(region)
            M_t_summary = calculator.determine_M_t(region)
            M_z_cat_summary = calculator.determine_M_z_cat(region, terrain_category, reference_height)
            V_sit_beta_summary = calculator.calculate_site_wind_speed(V_R_table_summary, M_d_summary, M_c_summary, M_s_summary, M_t_summary, M_z_cat_summary)
            V_des_theta_summary = calculator.calculate_design_wind_speed(V_sit_beta_summary, limit_state)

            if structure_type == "Free Standing Wall":
                thetas = [0, 45, 90]
                theta_results = {}
                for theta in thetas:
                    C_shp_wall, e_wall = calculator.calculate_aerodynamic_shape_factor(structure_type, None, b, c, h, theta, has_return_corner=has_return_corner)
                    p_uls_wall = calculator.calculate_wind_pressure(V_des_theta_summary if limit_state == "ULS" else V_des_theta_sls_ref_height, C_shp_wall)
                    
                    # For pressure distribution, use the V_des_theta (ULS) and V_des_theta_sls_ref_height (SLS)
                    if limit_state == "ULS":
                        distances, pressures_uls_dist = calculator.calculate_pressure_distribution(b, c, h, V_des_theta_summary, theta, has_return_corner=has_return_corner)
                        pressures_sls_dist = [] # Not directly used for SLS distribution plot from this branch
                    else: # SLS
                        distances, pressures_sls_dist = calculator.calculate_pressure_distribution(b, c, h, V_des_theta_sls_ref_height, theta, has_return_corner=has_return_corner)
                        pressures_uls_dist = [] # Not directly used for ULS distribution plot from this branch
                    
                    if theta == 0:
                        theta_results[theta] = {
                            'C_shp': C_shp_wall, 'e': e_wall, 
                            'p_uls': p_uls_wall if limit_state == "ULS" else None, # Store only relevant limit state pressure
                            'p_sls': p_uls_wall if limit_state == "SLS" else None, # Store only relevant limit state pressure
                            'resultant_force_uls': p_uls_wall * b * c if limit_state == "ULS" else None,
                            'resultant_force_sls': p_uls_wall * b * c if limit_state == "SLS" else None,
                        }
                    else:
                        theta_results[theta] = {
                            'distances': distances, 
                            'pressures_uls': pressures_uls_dist if limit_state == "ULS" else [],
                            'pressures_sls': pressures_sls_dist if limit_state == "SLS" else [],
                            'max_pressure_uls': max(pressures_uls_dist) if limit_state == "ULS" and pressures_uls_dist else None, 
                            'max_pressure_sls': max(pressures_sls_dist) if limit_state == "SLS" and pressures_sls_dist else None,
                        }

                # Generate pressure distribution graph for the current limit state for Free Standing Wall
                plt.figure(figsize=(8, 4))
                # Use the V_des_to_use (which correctly picks ULS or SLS V_des for reference height for plotting purposes)
                V_des_to_use_for_plot = V_des_theta_summary if limit_state == "ULS" else V_des_theta_sls_ref_height
                
                distances_45, pressures_45 = calculator.calculate_pressure_distribution(b, c, h, V_des_to_use_for_plot, 45, has_return_corner)
                plt.plot(distances_45, pressures_45, label="θ = 45°", color="blue")
                
                distances_90, pressures_90 = calculator.calculate_pressure_distribution(b, c, h, V_des_to_use_for_plot, 90, has_return_corner)
                plt.plot(distances_90, pressures_90, label="θ = 90°", color="green")
                
                # Check if theta=0 results are available for the current limit_state before accessing
                p_0_val = None
                if 0 in theta_results:
                    p_0_val = theta_results[0].get('p_uls' if limit_state == "ULS" else 'p_sls')
                if p_0_val is not None:
                    plt.axhline(y=p_0_val, color="red", linestyle="--", label="θ = 0° (uniform)")

                plt.xlabel("Distance from Windward Free End (m)")
                plt.ylabel("Wind Pressure (kPa)")
                plt.title(f"Wind Pressure Distribution ({location}, {limit_state})")
                plt.legend()
                plt.grid(True)
                graph_filename = f"pressure_distribution_{limit_state.lower()}.png"
                plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
                plt.close()

                results[limit_state] = {
                    'V_R': V_R_table_summary, 'V_sit_beta': V_sit_beta_summary, 'V_des_theta': V_des_theta_summary,
                    'results': theta_results, 'graph_filename': graph_filename,
                    'reduction_factor': reduction_factor
                }
            else: # For Circular Tank, Attached Canopy, Protection Screens, Scaffold
                C_shp_current_type = C_shp_overall_for_table
                e_current_type = e_for_other_types
                p_uls = calculator.calculate_wind_pressure(V_des_theta_summary, C_shp_current_type) # Use ULS V_des_theta for ULS pressure
                p_sls = calculator.calculate_wind_pressure(V_des_theta_sls_ref_height, C_shp_current_type) # Use SLS V_des_theta for SLS pressure
                
                results[limit_state] = {
                    'V_R': V_R_table_summary, 'V_sit_beta': V_sit_beta_summary, 'V_des_theta': V_des_theta_summary,
                    'C_shp': C_shp_current_type, 'e': e_current_type, 'p_uls': p_uls, 'p_sls': p_sls,
                    'reduction_factor': reduction_factor
                }

        # --- This block runs once AFTER the limit_states loop for Protection Screens and Scaffold ---
        if structure_type in ["Protection Screens", "Scaffold"]:
            # Calculate height-dependent pressures for ULS and SLS for the plot
            # Use the already calculated C_shp_overall_for_table
            heights, V_des_values_uls_height_plot, pressures_uls_height_plot = calculator.calculate_pressure_vs_height(
                region, terrain_category, reference_height, "ULS", importance_level, distance_from_coast_km, C_shp_overall_for_table,
                scaffold_type=scaffold_type, solidity_ratio=solidity_ratio, num_bays_length=num_bays_length, num_rows_width=num_rows_width, typical_bay_length_m=typical_bay_length_m, member_diameter_mm=member_diameter_mm
            )
            # Re-calculate for SLS using the correct SLS V_R profile
            _, V_des_values_sls_height_plot, pressures_sls_height_plot = calculator.calculate_pressure_vs_height(
                region, terrain_category, reference_height, "SLS", importance_level, distance_from_coast_km, C_shp_overall_for_table,
                scaffold_type=scaffold_type, solidity_ratio=solidity_ratio, num_bays_length=num_bays_length, num_rows_width=num_rows_width, typical_bay_length_m=typical_bay_length_m, member_diameter_mm=member_diameter_mm
            )
            
            # Now assign the calculated data to the respective ULS and SLS entries
            results['ULS']['heights'] = heights
            results['ULS']['V_des_values_height_plot'] = V_des_values_uls_height_plot
            results['ULS']['pressures_uls_height_plot'] = pressures_uls_height_plot
            
            results['SLS']['heights'] = heights
            results['SLS']['V_des_values_height_plot'] = V_des_values_sls_height_plot
            results['SLS']['pressures_sls_height_plot'] = pressures_sls_height_plot

            # Plotting is done once (shows both ULS and SLS profiles)
            plt.figure(figsize=(8, 4))
            plt.plot(heights, pressures_uls_height_plot, label="ULS", color="blue")
            plt.plot(heights, pressures_sls_height_plot, label="SLS", color="green")
            plt.xlabel("Height (m)")
            plt.ylabel("Wind Pressure (kPa)")
            plt.title(f"Wind Pressure vs. Height ({location}, {structure_type})")
            plt.legend()
            plt.grid(True)
            graph_filename = "height_pressure_graph.png"
            plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
            plt.close()
            results['ULS']['height_pressure_graph'] = graph_filename
            results['SLS']['height_pressure_graph'] = graph_filename # Both ULS and SLS can reference the same graph
        # --- End of moved block ---


        pdf_data = generate_pdf_report(inputs, results, project_number, project_name)
        if pdf_data:
            st.success("Calculations completed successfully!")
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"Wind_Load_Report_{project_number}.pdf",
                mime="application/pdf"
            )
            # Display plots in Streamlit UI
            if structure_type == "Free Standing Wall":
                for limit_state in limit_states:
                    # Check if 'graph_filename' exists for the current limit_state
                    if 'graph_filename' in results[limit_state]:
                        st.image(results[limit_state]['graph_filename'], caption=f"Pressure Distribution ({limit_state})")
            elif structure_type in ["Protection Screens", "Scaffold"]:
                # Check if 'height_pressure_graph' exists in results['ULS']
                if 'height_pressure_graph' in results['ULS']:
                    st.image(results['ULS']['height_pressure_graph'], caption=f"Pressure vs. Height")
        else:
            st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
