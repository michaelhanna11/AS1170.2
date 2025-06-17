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
    """
    Loads an image from a URL, with a fallback.
    Args:
        url (str): Primary URL for the image.
    Returns:
        PIL.Image.Image or None: Loaded image or None if loading fails.
    """
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
    """
    Calculates wind loads based on AS/NZS 1170.2:2021.
    """
    def __init__(self):
        """
        Initializes the calculator with standard-specific data tables.
        """
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
        # Reduction factors for construction duration (Table 3.2.6(B) of AS/NZS 1170.2:2021)
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
        """
        Determines the wind region based on the selected Australian location.
        Args:
            location (str): Australian city name.
        Returns:
            str: Wind region (e.g., "A2", "C").
        """
        region_map = {
            "Sydney": "A2", "Melbourne": "A4", "Brisbane": "B1", "Perth": "A1", "Adelaide": "A5",
            "Darwin": "C", "Cairns": "C", "Townsville": "B2", "Port Hedland": "D", "Alice Springs": "A0", "Hobart": "A4",
        }
        return region_map.get(location, "A2") # Default to Sydney's region if not found

    def interpolate_V_R(self, region, R, distance_from_coast_km):
        """
        Interpolates Regional Gust Wind Speed (V_R) for Regions C and D based on distance from coast.
        Ref: AS/NZS 1170.2:2021 Clause 3.2, Table 3.1(A)
        Args:
            region (str): Wind region.
            R (int): Average Recurrence Interval (years).
            distance_from_coast_km (float): Distance from coast in km.
        Returns:
            float: Interpolated V_R value.
        """
        if region not in self.regions_with_interpolation:
            return self.V_R_table[region][R]
        
        # Values are based on interpretation of interpolation clause.
        # Standard says linear interpolation between max value and reduced values at 100km, 200km.
        # The reduction factors 0.95 and 0.90 are assumed here as typical for such interpolation.
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

    def determine_V_R(self, region, limit_state, importance_level=None, distance_from_coast_km=None, construction_period=None, reference_height=None):
        """
        Determines the Regional Gust Wind Speed (V_R) for ULS or SLS.
        Ref: AS/NZS 1170.2:2021 Clause 3.2, Table 3.1(A) and related SLS guidance.
        Args:
            region (str): Wind region.
            limit_state (str): "ULS" or "SLS".
            importance_level (str, optional): "I", "II", or "III" for ULS.
            distance_from_coast_km (float, optional): Distance from coast for interpolation.
            construction_period (str, optional): Duration for reduction factor.
            reference_height (float, optional): Height for SLS V_R calculation.
        Returns:
            tuple: (V_R value, reduction_factor applied).
        """
        if limit_state == "SLS":
            # For SLS, AS/NZS 1170.2, Clause 3.2(1) refers to a mean wind speed.
            # A common approach (e.g., for serviceability deflection) is based on a mean wind speed profile.
            V_mean = 16.0  # Mean wind speed for SLS (indicative value based on common practice for this standard)
            if reference_height is not None:
                # This formula is V(z) = [(z/10)^0.14 + 0.4] * V_mean, a simplified profile often used for mean wind.
                # Here, we use it to derive a V_R for SLS purposes, representing a lower frequent wind speed.
                V_R = ((reference_height / 10) ** 0.14 + 0.4) * V_mean
            else:
                V_R = 16.0  # Fallback
            reduction_factor = 1.0  # No construction duration reduction for SLS
            return V_R, reduction_factor
        
        # ULS calculation
        R_map = {"I": 25, "II": 100, "III": 250} # Average Recurrence Interval (R) for Importance Levels (Table 3.1(A))
        if importance_level not in R_map:
            raise ValueError("Importance level must be 'I', 'II', or 'III' for ULS.")
        R = R_map[importance_level]
        
        # Base V_R from table or interpolated
        if region in self.regions_with_interpolation and distance_from_coast_km is not None:
            V_R = self.interpolate_V_R(region, R, distance_from_coast_km)
        else:
            V_R = self.V_R_table[region][R]
        
        # Apply construction duration reduction factor (Table 3.2.6(B))
        if construction_period:
            region_category = self.region_to_category.get(region, "A")
            reduction_factor = self.reduction_factors[region_category].get(construction_period, 1.0)
            V_R = V_R * reduction_factor
        else:
            reduction_factor = 1.0
        
        return V_R, reduction_factor

    def determine_M_c(self, region):
        """
        Determines the Climate Change Multiplier (M_c).
        Ref: AS/NZS 1170.2:2021 Clause 3.4, Table 3.3.
        Args:
            region (str): Wind region.
        Returns:
            float: M_c value.
        """
        return self.M_c_table[region]

    def determine_M_d(self, region):
        """
        Determines the Wind Direction Multiplier (M_d).
        Ref: AS/NZS 1170.2:2021 Clause 3.3, Table 3.2(A).
        Note: For simplicity, a value of 1.0 is used here, as applying full M_d
        requires directional analysis and is often simplified for general purpose tools.
        For specific designs, values from Table 3.2(A) or (B) should be used.
        Args:
            region (str): Wind region.
        Returns:
            float: M_d value.
        """
        return 1.0

    def determine_M_s(self, region):
        """
        Determines the Shielding Multiplier (M_s).
        Ref: AS/NZS 1170.2:2021 Clause 4.3.
        Note: For simplicity, a value of 1.0 is used as detailed shielding calculations
        require complex input (e.g., surrounding building data).
        Args:
            region (str): Wind region.
        Returns:
            float: M_s value.
        """
        return 1.0

    def determine_M_t(self, region):
        """
        Determines the Topographic Multiplier (M_t).
        Ref: AS/NZS 1170.2:2021 Clause 4.4.
        Note: For simplicity, a value of 1.0 is used as detailed topographic calculations
        require specific hill/escarpment geometry.
        Args:
            region (str): Wind region.
        Returns:
            float: M_t value.
        """
        return 1.0

    def determine_M_z_cat(self, region, terrain_category, height):
        """
        Determines the Terrain/Height Multiplier (M_z,cat).
        Ref: AS/NZS 1170.2:2021 Clause 4.2.2, Table 4.1.
        Args:
            region (str): Wind region.
            terrain_category (str): Terrain category (e.g., "TC2").
            height (float): Height above ground in meters.
        Returns:
            float: M_z,cat value.
        """
        # Special rule for Region A0 for heights > 100m.
        if region == "A0" and height > 100:
            return 1.24 if height <= 200 else 1.24 # Fixed value for this range in A0.
        
        terrain_data = self.M_z_cat_table[terrain_category]
        heights = sorted(terrain_data.keys())

        # Direct hit
        if height in heights:
            return terrain_data[height]
        
        # Below lowest tabulated height
        if height <= heights[0]:
            return terrain_data[heights[0]]
        
        # Above highest tabulated height
        if height >= heights[-1]:
            return terrain_data[heights[-1]]
        
        # Linear interpolation for intermediate heights
        for i in range(len(heights) - 1):
            h1, h2 = heights[i], heights[i + 1]
            if h1 < height <= h2:
                m1, m2 = terrain_data[h1], terrain_data[h2]
                fraction = (height - h1) / (h2 - h1)
                return m1 + fraction * (m2 - m1)
        return 1.0 # Fallback

    def calculate_site_wind_speed(self, V_R, M_d, M_c, M_s, M_t, M_z_cat):
        """
        Calculates the Site Wind Speed (V_sit,beta).
        Ref: AS/NZS 1170.2:2021 Equation 2.2.
        Args:
            V_R (float): Regional Gust Wind Speed.
            M_d (float): Wind Direction Multiplier.
            M_c (float): Climate Change Multiplier.
            M_s (float): Shielding Multiplier.
            M_t (float): Topographic Multiplier.
            M_z_cat (float): Terrain/Height Multiplier.
        Returns:
            float: Site Wind Speed.
        """
        return V_R * M_c * M_d * M_z_cat * M_s * M_t

    def calculate_design_wind_speed(self, V_sit_beta, limit_state):
        """
        Calculates the Design Wind Speed (V_des,theta).
        Ref: AS/NZS 1170.2:2021 Clause 2.3.
        Args:
            V_sit_beta (float): Site Wind Speed.
            limit_state (str): "ULS" or "SLS".
        Returns:
            float: Design Wind Speed.
        """
        # For ultimate limit states design, V_des,theta shall not be less than 30 m/s.
        if limit_state == "ULS":
            return max(V_sit_beta, 30.0)
        return V_sit_beta

    def calculate_Cpn_freestanding_wall(self, b, c, h, theta, solidity_ratio, distance_from_windward_end=None, has_return_corner=False):
        """
        Calculates Net Pressure Coefficient (Cpn) for freestanding walls, then applies K_p for C_shp.
        Ref: AS/NZS 1170.2:2021 Appendix B, Tables B.2(A), B.2(B), B.2(C), B.2(D), Clause B.1.4.
        Args:
            b (float): Width of the wall (b, along wind direction for theta=0).
            c (float): Height of the wall.
            h (float): Reference height (total height of structure).
            theta (int): Wind direction (0, 45, or 90 degrees).
            solidity_ratio (float): Solidity ratio (δ) of the wall.
            distance_from_windward_end (float, optional): Required for theta=45, 90.
            has_return_corner (bool): For 45 deg wind, if return corner extends > 1c.
        Returns:
            tuple: (C_shp value, eccentricity e).
        """
        b_over_c = b / c
        c_over_h = c / h
        
        _Cpn_unadjusted = 0.0 # Initialize unadjusted Cpn
        e = 0.0 # Initialize eccentricity

        if theta == 0: # Wind normal to hoarding or wall (Table B.2(A))
            if 0.5 <= b_over_c <= 5:
                if 0.2 <= c_over_h <= 1:
                    _Cpn_unadjusted = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else: # c/h < 0.2 (common for long, low walls)
                    _Cpn_unadjusted = 1.4 + 0.3 * log10(b_over_c)
            else: # b/c > 5
                if 0.2 <= c_over_h <= 1:
                    _Cpn_unadjusted = 1.7 - 0.5 * c_over_h
                else: # c/h < 0.2 (common for very long, very low walls)
                    _Cpn_unadjusted = 1.4 + 0.3 * log10(b_over_c) # For all b/c when c/h < 0.2
            e = 0.0
        elif theta == 45: # Wind at 45 degrees to hoarding or wall (Tables B.2(B) & B.2(C))
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=45°.")
            
            if 0.5 <= b_over_c <= 5: # Table B.2(B)
                if 0.2 <= c_over_h <= 1:
                    _Cpn_unadjusted = 1.3 + 0.5 * (0.3 + log10(b_over_c)) * (0.8 - c_over_h)
                else: # c/h < 0.2
                    _Cpn_unadjusted = 1.4 + 0.3 * log10(b_over_c)
            else: # b/c > 5 (Table B.2(C))
                if c_over_h <= 0.7:
                    if distance_from_windward_end <= 2 * c:
                        _Cpn_unadjusted = 3.0
                    elif distance_from_windward_end <= 4 * c:
                        _Cpn_unadjusted = 1.5
                    else:
                        _Cpn_unadjusted = 0.75
                else: # c/h > 0.7
                    if distance_from_windward_end <= 2 * h: # 'h' is total height (reference_height)
                        _Cpn_unadjusted = 2.4
                    elif distance_from_windward_end <= 4 * h:
                        _Cpn_unadjusted = 1.2
                    else:
                        _Cpn_unadjusted = 0.6
                
                # Apply return corner condition (note in Table B.2(C))
                if has_return_corner:
                    if c_over_h <= 0.7 and distance_from_windward_end <= 2 * c:
                        _Cpn_unadjusted = 2.2 # Overrides 3.0
                    elif c_over_h > 0.7 and distance_from_windward_end <= 2 * h:
                        _Cpn_unadjusted = 1.8 # Overrides 2.4
            e = 0.2 * b # Eccentricity for 45 deg wind.
        elif theta == 90: # Wind parallel to hoarding or wall (Table B.2(D))
            if distance_from_windward_end is None:
                raise ValueError("Distance required for theta=90°.")
            
            if c_over_h <= 0.7:
                if distance_from_windward_end <= 2 * c:
                    _Cpn_unadjusted = 1.2
                elif distance_from_windward_end <= 4 * c:
                    _Cpn_unadjusted = 0.6
                else:
                    _Cpn_unadjusted = 0.3
            else: # c/h > 0.7
                if distance_from_windward_end <= 2 * h: # 'h' is total height (reference_height)
                    _Cpn_unadjusted = 1.0
                elif distance_from_windward_end <= 4 * h:
                    _Cpn_unadjusted = 0.25
                else:
                    _Cpn_unadjusted = 0.25
            _Cpn_unadjusted = abs(_Cpn_unadjusted) # Table B.2(D) values are +/-. For calculation, use absolute.
            e = 0.0 # Eccentricity for 90 deg wind.
        else:
            raise ValueError("Theta must be 0°, 45°, or 90°.")

        # Apply Net Porosity Factor (K_p) - Clause B.1.4
        K_p = 1 - (1 - solidity_ratio)**2 # K_p = 1 - (1 - delta)^2
        C_shp = _Cpn_unadjusted * K_p
        
        return C_shp, e

    def _calculate_Cshp_open_scaffold(self, solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta, reference_height, typical_bay_width_m):
        """
        Calculates Aerodynamic Shape Factor (Cshp) for an open (unclad) scaffold.
        Ref: AS/NZS 1170.2:2021 Appendix C (Lattice towers), especially C.2.2, C.2.3, Table C.6(B), Table C.2.
        Assumptions:
            - Circular members (steel tubes).
            - Wind normal to the face (most critical for overall drag).
            - Flow regime (sub/super-critical) is determined dynamically.
            - Lambda for K_sh interpolation calculated dynamically from bay dimensions.
        Args:
            solidity_ratio (float): User-provided overall solidity ratio (delta).
            num_bays_length (int): Number of bays along the scaffold length.
            num_rows_width (int): Number of rows perpendicular to the wind.
            typical_bay_length_m (float): Length of a typical bay (used as spacing for lambda).
            member_diameter_mm (float): Diameter of individual members (for bi*Vdes,theta).
            V_des_theta (float): Design wind speed (m/s) at reference height.
            reference_height (float): Total height of the scaffold.
            typical_bay_width_m (float): Width of a typical bay.
        Returns:
            float: Calculated Cshp for the open scaffold.
        """
        delta = solidity_ratio
        
        # 1. Effective solidity (delta_e) - Clause C.2.2, for circular members.
        delta_e = 1.2 * (delta ** 1.75) 

        # 2. Drag force coefficient (Cd) for a single frame - Table C.6(B) for circular members.
        member_diameter_m = member_diameter_mm / 1000.0 # Convert mm to m
        bi_Vdes_theta = member_diameter_m * V_des_theta

        # Define Cd lookup tables for both flow regimes (onto corner values, more conservative)
        solidity_points_cd = [0.0, 0.05, 0.1, 0.2, 0.3, 1.0] # Extended to 0 and 1 for interp

        # Onto corner values for sub-critical flow (bi*Vdes_theta < 3 m^2/s) from Table C.6(B)
        cd_sub_critical_onto_corner = [2.5, 2.5, 2.3, 2.3, 2.3, 2.3] 

        # Onto corner values for super-critical flow (bi*Vdes_theta >= 6 m^2/s) from Table C.6(B)
        cd_super_critical_onto_corner = [1.6, 1.6, 1.6, 1.7, 1.9, 1.9] 

        Cd_single_frame = 0.0
        if bi_Vdes_theta < 3.0: # Sub-critical flow
            Cd_single_frame = np.interp(delta, solidity_points_cd, cd_sub_critical_onto_corner)
        elif bi_Vdes_theta >= 6.0: # Super-critical flow
            Cd_single_frame = np.interp(delta, solidity_points_cd, cd_super_critical_onto_corner)
        else: # Transition flow (3.0 <= bi_Vdes_theta < 6.0), interpolate between critical and super-critical values
            Cd_at_3 = np.interp(delta, solidity_points_cd, cd_sub_critical_onto_corner)
            Cd_at_6 = np.interp(delta, solidity_points_cd, cd_super_critical_onto_corner)
            Cd_single_frame = np.interp(bi_Vdes_theta, [3.0, 6.0], [Cd_at_3, Cd_at_6])

        # Clamp Cd_single_frame to plausible range from the table (max is 2.5 from sub-critical, max is 1.9 from super-critical)
        Cd_single_frame = max(0.0, min(Cd_single_frame, 2.5)) 

        # 3. Shielding factor (K_sh) - Table C.2 for multiple frames.
        # Lambda (spacing ratio) = frame spacing / (smaller of l or b of frame) - Clause C.2.3 definition for lambda.
        # Here, frame spacing = typical_bay_length_m.
        # 'l' (vertical frame dimension) = reference_height (total scaffold height).
        # 'b' (horizontal frame dimension) = typical_bay_width_m.
        frame_min_dimension = min(reference_height, typical_bay_width_m)
        if frame_min_dimension <= 0: # Avoid division by zero if dimensions are invalid or zero.
            lambda_val = 1.0 # Default to 1.0 if invalid dimensions.
        else:
            lambda_val = typical_bay_length_m / frame_min_dimension
        
        # Clamp lambda_val to the range of Table C.2 for interpolation
        # Table C.2 lambda points are: <=0.2, 0.5, 1.0, 2.0, 4.0, >=8.0
        # We need to map these to numerical values for np.interp
        lambda_points_ksh = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 1000.0] # Extended points for interpolation
        
        # Data structure for Ksh based on delta_e rows and lambda columns from Table C.2 (Angle 0 deg)
        # Each inner list corresponds to a lambda_point: [lambda=0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 1000.0]
        # (Using first value for lambda <= 0.2 and last for lambda >= 8.0)
        ksh_table_data = {
            # delta_e: [ksh_at_lambda_0.0, ksh_at_lambda_0.2, ksh_at_lambda_0.5, ksh_at_lambda_1.0, ksh_at_lambda_2.0, ksh_at_lambda_4.0, ksh_at_lambda_8.0, ksh_at_lambda_inf]
            0.0:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Extrapolated/assumed for delta_e=0.0
            0.05: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Extrapolated/assumed for delta_e=0.05
            0.1:  [0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            0.2:  [0.5, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0], 
            0.3:  [0.3, 0.3, 0.6, 0.7, 0.7, 0.8, 1.0, 1.0], 
            0.4:  [0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 1.0, 1.0], 
            0.5:  [0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0], 
            0.7:  [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.9, 1.0], 
            1.0:  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 1.0] 
        }
        
        # Prepare data for 2D interpolation: x_points (delta_e), y_points (lambda_points), values (Ksh table)
        solidity_e_points = sorted(ksh_table_data.keys())
        
        # Interpolate Ksh for each effective solidity point across lambda values to get Ksh for our lambda_val
        ksh_values_for_our_lambda = []
        for d_e_point in solidity_e_points:
            ksh_vals_for_this_delta_e = ksh_table_data[d_e_point]
            K_sh_interp_lambda = np.interp(lambda_val, lambda_points_ksh, ksh_vals_for_this_delta_e)
            ksh_values_for_our_lambda.append(K_sh_interp_lambda)

        # Then, interpolate over delta_e using the Ksh values for our lambda_val
        K_sh_interpolated = np.interp(delta_e, solidity_e_points, ksh_values_for_our_lambda)
        
        # Clamp K_sh_interpolated to the valid range [0.2, 1.0] from Table C.2.
        K_sh_interpolated = max(0.2, min(K_sh_interpolated, 1.0))

        # 4. Total C_shp calculation - Clause C.2.3, Equation C.2(5)
        # C_shp = C_shp,1 + SUM(K_sh * C_shp,1)
        # This simplifies to C_shp,1 * (1 + SUM(K_sh)) = Cd_single_frame * (1 + sum_Ksh)
        # Sum is over subsequent downwind frames. If num_bays_length is 1 (single bay), sum is 0.
        
        sum_Ksh = 0.0
        if num_bays_length > 1:
            # Assuming all subsequent frames apply the same K_sh_interpolated value
            sum_Ksh = (num_bays_length - 1) * K_sh_interpolated 
        
        # The overall C_shp for the scaffold, considering multiple rows (width-wise).
        # num_rows_width multiplies the effective frontal area.
        C_shp_overall = Cd_single_frame * (1 + sum_Ksh) * num_rows_width 

        return C_shp_overall

    def calculate_aerodynamic_shape_factor(self, structure_type, user_C_shp=None, b=None, c=None, h=None, theta=None, distance_from_windward_end=None, has_return_corner=False, solidity_ratio=None, scaffold_type=None, num_bays_length=None, num_rows_width=None, typical_bay_length_m=None, typical_bay_width_m=None, member_diameter_mm=None, V_des_theta=None):
        """
        Determines the Aerodynamic Shape Factor (C_shp) for various structure types.
        Ref: AS/NZS 1170.2:2021 Section 5 and Appendix B, C.
        Args:
            structure_type (str): Type of structure.
            user_C_shp (float, optional): User-defined C_shp for Protection Screens.
            b (float, optional): Width of Free Standing Wall.
            c (float, optional): Height of Free Standing Wall.
            h (float, optional): Reference height (total structure height).
            theta (int, optional): Wind direction for Free Standing Wall.
            distance_from_windward_end (float, optional): For Free Standing Wall (theta=45, 90).
            has_return_corner (bool, optional): For Free Standing Wall (theta=45).
            solidity_ratio (float, optional): Solidity for Free Standing Wall or Open Scaffold.
            scaffold_type (str, optional): "Open (Unclad)" or "Fully Clad" for Scaffold.
            num_bays_length (int, optional): Bays in length for Scaffold.
            num_rows_width (int, optional): Rows in width for Scaffold.
            typical_bay_length_m (float, optional): Typical bay length for Scaffold.
            typical_bay_width_m (float, optional): Typical bay width for Scaffold.
            member_diameter_mm (float, optional): Member diameter for Open Scaffold.
            V_des_theta (float, optional): Design wind speed (m/s) for current height.
        Returns:
            tuple: (C_shp value, eccentricity e).
        """
        e = 0.0 # Default eccentricity to 0.0

        if structure_type == "Free Standing Wall":
            # Pass solidity_ratio to calculate_Cpn_freestanding_wall
            C_shp, e = self.calculate_Cpn_freestanding_wall(b, c, h, theta, solidity_ratio, distance_from_windward_end, has_return_corner)
            return C_shp, e
        elif structure_type == "Circular Tank":
            # Overall drag coefficient for circular sections, typically around 0.6-0.8 (Table 5.3(A) in old standards, or similar)
            return 0.8, e
        elif structure_type == "Attached Canopy":
            # For attached canopies, a C_shp of 1.2 is a common conservative value (Table B.9)
            return 1.2, e
        elif structure_type == "Protection Screens":
            if user_C_shp is None:
                raise ValueError("C_shp required for Protection Screens.")
            return user_C_shp, e
        elif structure_type == "Scaffold":
            if scaffold_type == "Open (Unclad)":
                if None in [solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta, h, typical_bay_width_m]:
                    raise ValueError("All scaffold parameters are required for Open (Unclad) Scaffold.")
                C_shp_scaffold = self._calculate_Cshp_open_scaffold(solidity_ratio, num_bays_length, num_rows_width, typical_bay_length_m, member_diameter_mm, V_des_theta, h, typical_bay_width_m)
                return C_shp_scaffold, e
            elif scaffold_type == "Fully Clad":
                # For a fully clad scaffold, treat it as a solid hoarding/wall.
                # A Cpn of ~1.2 (for long walls, Table B.2(A) where b/c > 5 and c/h = 1) is conservative.
                return 1.2, e
            else:
                raise ValueError("Invalid Scaffold type selected.")
        else:
            raise ValueError("Invalid structure type.")

    def calculate_wind_pressure(self, V_des_theta, C_shp):
        """
        Calculates the design wind pressure.
        Ref: AS/NZS 1170.2:2021 Equation 2.4(1).
        Args:
            V_des_theta (float): Design Wind Speed.
            C_shp (float): Aerodynamic Shape Factor.
        Returns:
            float: Design wind pressure in kPa.
        """
        rho_air = 1.2 # Density of air (kg/m^3)
        C_dyn = 1.0 # Dynamic response factor (default to 1.0 unless specific dynamic analysis is needed per Section 6)
        return (0.5 * rho_air) * (V_des_theta ** 2) * C_shp * C_dyn / 1000 # Convert Pa to kPa

    def calculate_pressure_distribution(self, b, c, h, V_des_theta, theta, solidity_ratio, has_return_corner=False):
        """
        Calculates wind pressure distribution along a Free Standing Wall for given theta.
        Args:
            b (float): Width of the wall.
            c (float): Height of the wall.
            h (float): Reference height (total structure height).
            V_des_theta (float): Design Wind Speed.
            theta (int): Wind direction.
            solidity_ratio (float): Solidity ratio of the wall.
            has_return_corner (bool): For 45 deg wind, if return corner extends > 1c.
        Returns:
            tuple: (distances, pressures) numpy arrays.
        """
        num_points = 100
        distances = np.linspace(0, b, num_points)
        pressures = []
        for d in distances:
            # C_shp is distance-dependent for theta=45, 90 deg.
            # Pass solidity_ratio to calculate_aerodynamic_shape_factor
            C_shp, _ = self.calculate_aerodynamic_shape_factor(
                "Free Standing Wall", None, b, c, h, theta, distance_from_windward_end=d, has_return_corner=has_return_corner, solidity_ratio=solidity_ratio
            )
            p = self.calculate_wind_pressure(V_des_theta, C_shp)
            pressures.append(p)
        return distances, pressures

    def calculate_pressure_vs_height(self, region, terrain_category, reference_height, limit_state, importance_level, distance_from_coast_km, C_shp_base, scaffold_type=None, solidity_ratio=None, num_bays_length=None, num_rows_width=None, typical_bay_length_m=None, member_diameter_mm=None, typical_bay_width_m=None):
        """
        Calculates wind pressure variation with height for structures where C_shp is constant with height
        (like Protection Screens, Tanks, Canopies, or a Simplified Scaffold model).
        Args:
            region (str): Wind region.
            terrain_category (str): Terrain category.
            reference_height (float): Total height of the structure.
            limit_state (str): "ULS" or "SLS".
            importance_level (str): Importance level for ULS.
            distance_from_coast_km (float): Distance from coast.
            C_shp_base (float): The aerodynamic shape factor at the reference height.
            scaffold_type (str, optional): "Open (Unclad)" or "Fully Clad" for Scaffold.
            solidity_ratio (float, optional): Solidity for Open Scaffold.
            num_bays_length (int, optional): Bays in length for Scaffold.
            num_rows_width (int, optional): Rows in width for Scaffold.
            typical_bay_length_m (float, optional): Typical bay length for Scaffold.
            member_diameter_mm (float, optional): Member diameter for Open Scaffold.
            typical_bay_width_m (float, optional): Typical bay width for Scaffold.
        Returns:
            tuple: (heights, V_des_values, pressures) numpy arrays showing variation with height.
        """
        height_step = 5.0
        heights = np.arange(0, reference_height + height_step, height_step)
        
        # Ensure that reference_height is exactly included and heights are unique and sorted.
        if reference_height not in heights:
            heights = np.append(heights, reference_height)
        heights = np.unique(np.sort(heights))

        V_des_values = []
        pressures = []
        
        # V_R_base is calculated once based on the overall reference_height for ULS.
        # For SLS, V_R is height-dependent.
        V_R_uls_base, _ = self.determine_V_R(region, "ULS", importance_level, distance_from_coast_km, reference_height=reference_height)
        
        for h_current in heights:
            if limit_state == "SLS":
                # V_R for SLS is derived from a mean wind speed profile, so it depends on current height.
                V_R_h_current, _ = self.determine_V_R(region, limit_state, reference_height=h_current)
            else: # ULS V_R does not typically change with height (if M_d, M_c etc. are constant)
                V_R_h_current = V_R_uls_base # Use the V_R calculated for the overall structure at ref height

            M_d = self.determine_M_d(region)
            M_c = self.determine_M_c(region)
            M_s = self.determine_M_s(region)
            M_t = self.determine_M_t(region)
            M_z_cat = self.determine_M_z_cat(region, terrain_category, h_current) # M_z,cat is always height-dependent
            
            V_sit_beta_current = self.calculate_site_wind_speed(V_R_h_current, M_d, M_c, M_s, M_t, M_z_cat)
            V_des_current = self.calculate_design_wind_speed(V_sit_beta_current, limit_state)

            # C_shp for these types (Protection Screens, Scaffold, etc.) are generally assumed constant with height,
            # using the C_shp_base calculated at the reference height.
            # However, for an Open (Unclad) Scaffold, the C_shp calculation depends on V_des_theta (for flow regime)
            # and height (for lambda calculation if reference_height is used as frame dimension).
            # So, re-calculate C_shp for Open (Unclad) Scaffold at each height.
            C_shp_at_current_height = C_shp_base # Default to the C_shp calculated at reference height
            if scaffold_type == "Open (Unclad)":
                C_shp_at_current_height, _ = self.calculate_aerodynamic_shape_factor( # The _ here is a dummy for eccentricity
                    "Scaffold", 
                    scaffold_type=scaffold_type, 
                    solidity_ratio=solidity_ratio, 
                    num_bays_length=num_bays_length, 
                    num_rows_width=num_rows_width, 
                    typical_bay_length_m=typical_bay_length_m, 
                    member_diameter_mm=member_diameter_mm,
                    V_des_theta=V_des_current, # Use the V_des_current for flow regime check
                    h=h_current, # Use current height for lambda calculation
                    typical_bay_width_m=typical_bay_width_m
                )

            p = self.calculate_wind_pressure(V_des_current, C_shp_at_current_height)
            V_des_values.append(V_des_current)
            pressures.append(p)
        return heights, V_des_values, pressures

# PDF generation functions
def build_elements(inputs, results, project_number, project_name):
    """
    Builds the list of ReportLab elements for the PDF report.
    Args:
        inputs (dict): Dictionary of user inputs.
        results (dict): Dictionary of calculation results.
        project_number (str): Project number.
        project_name (str): Project name.
    Returns:
        list: List of ReportLab flowables.
    """
    styles = getSampleStyleSheet()
    # Define custom paragraph styles for consistent formatting
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=16, spaceAfter=6, alignment=1)
    subtitle_style = ParagraphStyle(name='SubtitleStyle', parent=styles['Normal'], fontSize=10, spaceAfter=6, alignment=1)
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=12, spaceAfter=4)
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4, fontName='Helvetica-Bold')
    justified_style = ParagraphStyle(name='JustifiedStyle', parent=styles['Normal'], fontSize=9, spaceAfter=4, alignment=TA_JUSTIFY)
    
    # Table styles for headers and cells
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

    # Attempt to load logo from URL, falling back if necessary
    logo_file = "logo.png"
    try:
        response = requests.get(LOGO_URL, stream=True, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(logo_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception:
        try:
            response = requests.get(FALLBACK_LOGO_URL, stream=True, timeout=10)
            response.raise_for_status()
            with open(logo_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            logo_file = None # If both fail, don't use a logo file

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
    has_return_corner_text = "Yes" if inputs.get('has_return_corner') else "No"
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
            ["Solidity Ratio (δ)", f"{inputs['solidity_ratio_wall']:.3f}"], # Added solidity for wall
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
            # For Free Standing Wall, we display uniform C_shp for theta=0 in table,
            # and distributions for theta=45,90 in graph.
            theta_data_0 = data['results'][0]
            elements.append(Paragraph(f"Wind Direction: θ = 0°", normal_style))
            if limit_state == "ULS":
                table_data = [
                    [
                        Paragraph("Aerodynamic Shape Factor (<i>C<sub>shp</sub></i>)", table_header_style),
                        Paragraph("Eccentricity (e, m)", table_header_style),
                        Paragraph("<b>Wind Pressure (p, kPa) (ULS)</b>", table_header_style),
                        Paragraph("Resultant Force (kN) (ULS)", table_header_style),
                    ],
                    [
                        Paragraph(f"{theta_data_0['C_shp']:.3f}", table_cell_style),
                        Paragraph(f"{theta_data_0['e']:.2f}", table_cell_style),
                        Paragraph(f"{theta_data_0['p_uls']:.3f}", table_cell_style),
                        Paragraph(f"{theta_data_0['resultant_force_uls']:.2f}", table_cell_style),
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
                        Paragraph(f"{theta_data_0['C_shp']:.3f}", table_cell_style),
                        Paragraph(f"{theta_data_0['e']:.2f}", table_cell_style),
                        Paragraph(f"{theta_data_0['p_sls']:.3f}", table_cell_style),
                        Paragraph(f"{theta_data_0['resultant_force_sls']:.2f}", table_cell_style),
                    ]
                ]
                result_table = Table(table_data, colWidths=[45*mm, 35*mm, 35*mm, 35*mm])
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
    """
    Generates the PDF report using ReportLab.
    Args:
        inputs (dict): Dictionary of user inputs.
        results (dict): Dictionary of calculation results.
        project_number (str): Project number.
        project_name (str): Project name.
    Returns:
        bytes: PDF content as bytes, or None if error.
    """
    try:
        # First pass to determine total pages for accurate footer numbering
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
        total_pages = temp_doc.page # Get total pages from the temp build

        # Second pass to build the actual PDF with correct total page count in footer
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
        elements = build_elements(inputs, results, project_number, project_name) # Re-build elements for final PDF

        def final_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            page_num = canvas.getPageNumber()
            footer_text = f"{PROGRAM} {PROGRAM_VERSION} | tekhne © | Page {page_num}/{total_pages}"
            canvas.drawCentredString(doc.pagesize[0] / 2.0, 8 * mm, footer_text)
            canvas.restoreState()

        doc.build(elements, onFirstPage=final_footer, onLaterPages=final_footer)
        pdf_buffer.seek(0) # Reset buffer position to the beginning
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Streamlit UI
def main():
    """
    Main function to run the Streamlit application.
    """
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
    # These inputs trigger an immediate re-run of the script when changed,
    # allowing conditional UI elements (like structure-specific inputs) to update.
    
    col1, col2 = st.columns(2)
    with col1:
        project_number = st.text_input("Project Number", value="PRJ-001")
    with col2:
        project_name = st.text_input("Project Name", value="Sample Project")
    
    construction_period = st.selectbox(
        "Construction Duration",
        ["1 week", "1 month", "6 months", "More than 6 months"],
        index=2  # Default to 6 months
    )
    st.markdown("---")

    st.subheader("Location")
    location = st.selectbox("Select Location", calculator.valid_locations, 
                             index=calculator.valid_locations.index("Sydney"))

    importance_level = st.selectbox("Importance Level for ULS", ["I", "II", "III"])

    st.subheader("Terrain Category")
    terrain_options = {f"{key} ({value['name']}): {value['desc']}": key for key, value in calculator.terrain_categories.items()}
    terrain_choice = st.selectbox("Select Terrain Category", list(terrain_options.keys()))
    terrain_category = calculator.terrain_categories[terrain_options[terrain_choice]]["name"]

    reference_height = st.number_input("Reference Height (m)", min_value=0.1, value=10.0, step=0.1)

    region = calculator.determine_wind_region(location)
    distance_from_coast_km = None
    if region in ["C", "D"]:
        distance_from_coast_km = st.number_input("Distance from Coast (km)", min_value=50.0, max_value=200.0, value=50.0, step=1.0)

    st.subheader("Structure Type")
    structure_choice = st.selectbox("Select Structure Type", list(calculator.structure_types.values()))
    structure_type = structure_choice

    # Initialize all structure-specific variables to None. They will be populated
    # conditionally based on `structure_type` selection.
    b = c = user_C_shp = length = width = solidity_ratio = num_bays_length = num_rows_width = typical_bay_length_m = typical_bay_width_m = member_diameter_mm = scaffold_type = solidity_ratio_wall = None
    has_return_corner = False # Specific to Free Standing Wall

    if structure_type == "Free Standing Wall":
        b = st.number_input("Width of the Wall (b, m)", min_value=0.1, value=10.0, step=0.1)
        c = st.number_input("Height of the Wall (c, m)", min_value=0.1, max_value=reference_height, value=min(3.0, reference_height), step=0.1)
        solidity_ratio_wall = st.number_input("Solidity Ratio (δ)", min_value=0.01, max_value=1.0, value=1.0, step=0.01, help="Ratio of solid area to total area. For solid walls, δ=1.0.")
        one_c = c
        st.write(f"Note: 1c = {one_c:.2f} m (based on wall height c)")
        has_return_corner = st.checkbox(f"Return Corner Extends More Than 1c ({one_c:.2f} m)")
    elif structure_type == "Protection Screens":
        user_C_shp = st.number_input("Aerodynamic Shape Factor (C_shp)", min_value=0.1, value=1.0, step=0.01)
    elif structure_type == "Scaffold":
        scaffold_type = st.selectbox("Scaffold Configuration", ["Open (Unclad)", "Fully Clad"])
        length = st.number_input("Overall Length (m)", min_value=1.0, value=10.0, step=0.1)
        width = st.number_input("Overall Width (m)", min_value=0.5, value=1.2, step=0.1) 
        
        if scaffold_type == "Open (Unclad)":
            solidity_ratio = st.number_input("Solidity Ratio (δ)", min_value=0.01, max_value=0.99, value=0.15, step=0.01, help="Ratio of solid area to total area. Typical range: 0.1 to 0.3 for open scaffolds.")
            num_bays_length = st.number_input("Number of Bays (Length-wise)", min_value=1, value=4, step=1, help="Number of bays along the scaffold length.")
            num_rows_width = st.number_input("Number of Rows (Width-wise)", min_value=1, value=2, step=1, help="Number of rows perpendicular to the wind.")
            typical_bay_length_m = st.number_input("Typical Bay Length (m)", min_value=0.5, value=2.4, step=0.1, help="Length of a single bay (e.g., 2.4m, 1.8m). Used for shielding factor calculation.")
            typical_bay_width_m = st.number_input("Typical Bay Width (m)", min_value=0.5, value=1.2, step=0.1, help="Width of a single bay. Used in overall dimensions.")
            member_diameter_mm = st.number_input("Typical Member Diameter (mm)", min_value=10.0, value=48.3, step=0.1, help="Typical outer diameter of scaffold tubes (e.g., 48.3mm for steel).")

    # --- End of inputs moved outside the form ---

    # The form contains only the submit button to trigger calculations and PDF generation.
    with st.form(key='wind_load_form'):
        submit_button = st.form_submit_button(label="Calculate and Generate Report")

    if submit_button:
        # Collect all inputs into a dictionary for easy passing to PDF function
        inputs_for_calc = {
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
            'solidity_ratio_wall': solidity_ratio_wall, # Passed for Free Standing Wall
            'user_C_shp': user_C_shp, 
            'scaffold_type': scaffold_type, 
            'length': length, 
            'width': width, 
            'solidity_ratio': solidity_ratio, 
            'num_bays_length': num_bays_length, 
            'num_rows_width': num_rows_width, 
            'typical_bay_length_m': typical_bay_length_m, 
            'typical_bay_width_m': typical_bay_width_m, 
            'member_diameter_mm': member_diameter_mm, 
            'construction_period': construction_period
        }

        # --- Initial Calculations for Reference Height (used in tables and C_shp derivation) ---
        # Calculate SLS wind speed separately for service wind pressure at reference height
        V_R_sls_ref_height, _ = calculator.determine_V_R(
            region, "SLS", reference_height=reference_height
        )
        M_d_ref = calculator.determine_M_d(region)
        M_c_ref = calculator.determine_M_c(region)
        M_s_ref = calculator.determine_M_s(region)
        M_t_ref = calculator.determine_M_t(region)
        M_z_cat_ref_height = calculator.determine_M_z_cat(region, terrain_category, reference_height)
        V_sit_beta_sls_ref_height = calculator.calculate_site_wind_speed(V_R_sls_ref_height, M_d_ref, M_c_ref, M_s_ref, M_t_ref, M_z_cat_ref_height)
        V_des_theta_sls_ref_height = calculator.calculate_design_wind_speed(V_sit_beta_sls_ref_height, "SLS")

        # Calculate ULS V_des_theta at reference height, needed for some C_shp derivations (e.g. scaffold flow regime)
        V_R_uls_ref_height, reduction_factor_uls = calculator.determine_V_R(
            region, "ULS", importance_level, distance_from_coast_km, construction_period, reference_height
        )
        V_sit_beta_uls_ref_height = calculator.calculate_site_wind_speed(V_R_uls_ref_height, M_d_ref, M_c_ref, M_s_ref, M_t_ref, M_z_cat_ref_height)
        V_des_theta_uls_ref_height = calculator.calculate_design_wind_speed(V_sit_beta_uls_ref_height, "ULS")
        # --- End Initial Calculations ---


        limit_states = ["ULS", "SLS"]
        results = {}
        
        # Determine the C_shp and eccentricity that will be used for calculation.
        # This needs to happen ONCE for structure types other than Free Standing Wall,
        # using the V_des_theta at the reference height.
        C_shp_overall_for_table = 0.0 # Initialize to a safe default
        e_for_other_types = 0.0 # Default eccentricity to 0.0

        if structure_type == "Free Standing Wall":
            # Pass solidity_ratio_wall to calculate_aerodynamic_shape_factor
            # For the table, we always show theta=0's C_shp
            C_shp_overall_for_table, e_for_other_types = calculator.calculate_aerodynamic_shape_factor(
                structure_type, b=b, c=c, h=reference_height, theta=0, solidity_ratio=solidity_ratio_wall, has_return_corner=has_return_corner
            )
        elif structure_type == "Protection Screens":
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
                V_des_theta=V_des_theta_uls_ref_height, # Use V_des_theta_uls_ref_height for flow regime determination
                h=reference_height # Pass reference_height for lambda calculation
            )
        elif structure_type in ["Circular Tank", "Attached Canopy"]:
             C_shp_overall_for_table, e_for_other_types = calculator.calculate_aerodynamic_shape_factor(structure_type)


        # --- Main Loop to Populate Results for ULS and SLS ---
        for limit_state in limit_states:
            # Re-calculate V_R, V_sit_beta, V_des_theta for the summary table (at reference height)
            # These values will go directly into the results dictionary for display
            V_R_summary, reduction_factor_summary = calculator.determine_V_R(
                region, limit_state, importance_level, distance_from_coast_km, construction_period, reference_height
            )
            M_d_summary = calculator.determine_M_d(region)
            M_c_summary = calculator.determine_M_c(region)
            M_s_summary = calculator.determine_M_s(region)
            M_t_summary = calculator.determine_M_t(region)
            M_z_cat_summary = calculator.determine_M_z_cat(region, terrain_category, reference_height)
            V_sit_beta_summary = calculator.calculate_site_wind_speed(V_R_summary, M_d_summary, M_c_summary, M_s_summary, M_t_summary, M_z_cat_summary)
            V_des_theta_summary = calculator.calculate_design_wind_speed(V_sit_beta_summary, limit_state)

            if structure_type == "Free Standing Wall":
                # For Free Standing Wall, we store results for theta=0 in theta_results
                # The plots will be generated separately using calculate_pressure_distribution
                
                # Calculate for theta = 0 (uniform pressure) for the summary table
                C_shp_0_deg, e_0_deg = calculator.calculate_Cpn_freestanding_wall(
                    b, c, reference_height, 0, solidity_ratio_wall, distance_from_windward_end=0 # Pass dummy 0 for distance, will be ignored by theta=0 logic
                )
                p_uls_0_deg = calculator.calculate_wind_pressure(V_des_theta_uls_ref_height, C_shp_0_deg)
                p_sls_0_deg = calculator.calculate_wind_pressure(V_des_theta_sls_ref_height, C_shp_0_deg)

                theta_results = {
                    0: {
                        'C_shp': C_shp_0_deg, 'e': e_0_deg, 
                        'p_uls': p_uls_0_deg, 
                        'p_sls': p_sls_0_deg,
                        'resultant_force_uls': p_uls_0_deg * b * c,
                        'resultant_force_sls': p_sls_0_deg * b * c,
                    }
                }

                # Generate pressure distribution graph for the current limit state for Free Standing Wall
                plt.figure(figsize=(8, 4))
                V_des_to_use_for_plot = V_des_theta_uls_ref_height if limit_state == "ULS" else V_des_theta_sls_ref_height
                
                # Plot for theta = 45 (distributed pressure)
                distances_45, pressures_45 = calculator.calculate_pressure_distribution(b, c, reference_height, V_des_to_use_for_plot, 45, solidity_ratio_wall, has_return_corner=has_return_corner)
                plt.plot(distances_45, pressures_45, label="θ = 45°", color="blue")
                
                # Plot for theta = 90 (distributed pressure)
                distances_90, pressures_90 = calculator.calculate_pressure_distribution(b, c, reference_height, V_des_to_use_for_plot, 90, solidity_ratio_wall, has_return_corner=has_return_corner)
                plt.plot(distances_90, pressures_90, label="θ = 90°", color="green")
                
                # Add uniform pressure for theta = 0
                plt.axhline(y=p_uls_0_deg if limit_state == "ULS" else p_sls_0_deg, color="red", linestyle="--", label="θ = 0° (uniform)")

                plt.xlabel("Distance from Windward Free End (m)")
                plt.ylabel("Wind Pressure (kPa)")
                plt.title(f"Wind Pressure Distribution ({location}, {limit_state})")
                plt.legend()
                plt.grid(True)
                graph_filename = f"pressure_distribution_{limit_state.lower()}.png"
                plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
                plt.close()

                results[limit_state] = {
                    'V_R': V_R_summary, 'V_sit_beta': V_sit_beta_summary, 'V_des_theta': V_des_theta_summary,
                    'results': theta_results, 'graph_filename': graph_filename,
                    'reduction_factor': reduction_factor_summary if limit_state == "ULS" else 1.0 # Only ULS has reduction
                }
            else: # For Circular Tank, Attached Canopy, Protection Screens, Scaffold
                # These types use the C_shp_overall_for_table determined once using ULS V_des_theta at ref height
                C_shp_current_type = C_shp_overall_for_table
                e_current_type = e_for_other_types
                
                # Calculate p_uls and p_sls using their respective V_des_theta at reference height
                p_uls = calculator.calculate_wind_pressure(V_des_theta_uls_ref_height, C_shp_current_type) 
                p_sls = calculator.calculate_wind_pressure(V_des_theta_sls_ref_height, C_shp_current_type) 
                
                results[limit_state] = {
                    'V_R': V_R_summary, 'V_sit_beta': V_sit_beta_summary, 'V_des_theta': V_des_theta_summary,
                    'C_shp': C_shp_current_type, 'e': e_current_type, 'p_uls': p_uls, 'p_sls': p_sls,
                    'reduction_factor': reduction_factor_summary if limit_state == "ULS" else 1.0
                }

        # --- This block runs once AFTER the limit_states loop for Protection Screens and Scaffold ---
        # It calculates height-dependent pressures and generates the combined ULS/SLS pressure-vs-height plot.
        if structure_type in ["Protection Screens", "Scaffold"]:
            # Pass all relevant inputs to calculate_pressure_vs_height for scaffold if it's a scaffold
            calc_pressure_vs_height_kwargs = {
                'region': region,
                'terrain_category': terrain_category,
                'reference_height': reference_height,
                'importance_level': importance_level,
                'distance_from_coast_km': distance_from_coast_km,
                'C_shp_base': C_shp_overall_for_table,
            }
            if structure_type == "Scaffold":
                calc_pressure_vs_height_kwargs.update({
                    'scaffold_type': scaffold_type,
                    'solidity_ratio': solidity_ratio,
                    'num_bays_length': num_bays_length,
                    'num_rows_width': num_rows_width,
                    'typical_bay_length_m': typical_bay_length_m,
                    'member_diameter_mm': member_diameter_mm,
                    'typical_bay_width_m': typical_bay_width_m # Added for lambda calculation
                })

            heights, V_des_values_uls_height_plot, pressures_uls_height_plot = calculator.calculate_pressure_vs_height(
                limit_state="ULS", **calc_pressure_vs_height_kwargs
            )
            _, V_des_values_sls_height_plot, pressures_sls_height_plot = calculator.calculate_pressure_vs_height(
                limit_state="SLS", **calc_pressure_vs_height_kwargs
            )
            
            # Store height data and pressures in results for both ULS and SLS for the PDF
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
            plt.savefig("height_pressure_graph.png", bbox_inches='tight', dpi=150)
            plt.close()
            results['ULS']['height_pressure_graph'] = "height_pressure_graph.png"
            results['SLS']['height_pressure_graph'] = "height_pressure_graph.png" # Both ULS and SLS can reference the same graph
        # --- End of moved block ---


        pdf_data = generate_pdf_report(inputs_for_calc, results, project_number, project_name)
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
                    if 'graph_filename' in results[limit_state]:
                        st.image(results[limit_state]['graph_filename'], caption=f"Pressure Distribution ({limit_state})")
            elif structure_type in ["Protection Screens", "Scaffold"]:
                if 'height_pressure_graph' in results['ULS']:
                    st.image(results['ULS']['height_pressure_graph'], caption=f"Pressure vs. Height")
        else:
            st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
