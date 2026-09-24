"""Curated Indian district centroids, for nearest-centroid district/state
labeling of hazard points (map labels + Twilio alerts need "which district"
not just raw lat/lon).

This is NOT an exhaustive census district-boundary shapefile (India has
~750 districts; no polygon geometry is used here) — it's ~130 major
district headquarters (state capitals, largest cities per state, at least
one entry per state/UT) with their real lat/lon, used as centroids for a
simple nearest-neighbor assignment: a hazard point gets labeled with
whichever centroid is closest. This is the same pragmatic
"approximate-but-documented" pattern the rest of this codebase uses
(REGIONS' fixed-size demo boxes, hazard_india's nearest-lightning-strike
proximity check) rather than pulling in a multi-MB boundary dataset for a
hackathon timescale. Good enough to say "this hail cell is nearest to
Nagpur, Maharashtra" — not good enough to be a legal administrative
boundary lookup.
"""
import numpy as np

# (district, state, lat, lon) — district headquarters coordinates.
DISTRICTS = [
    ("Mumbai City", "Maharashtra", 18.94, 72.84),
    ("Mumbai Suburban", "Maharashtra", 19.08, 72.88),
    ("Pune", "Maharashtra", 18.52, 73.86),
    ("Nagpur", "Maharashtra", 21.15, 79.09),
    ("Nashik", "Maharashtra", 19.99, 73.79),
    ("Thane", "Maharashtra", 19.22, 72.98),
    ("Chhatrapati Sambhajinagar", "Maharashtra", 19.88, 75.34),
    ("Solapur", "Maharashtra", 17.66, 75.91),
    ("Kolhapur", "Maharashtra", 16.70, 74.24),
    ("Amravati", "Maharashtra", 20.93, 77.75),
    ("New Delhi", "Delhi", 28.61, 77.21),
    ("North Delhi", "Delhi", 28.71, 77.19),
    ("South Delhi", "Delhi", 28.55, 77.20),
    ("Lucknow", "Uttar Pradesh", 26.85, 80.95),
    ("Kanpur Nagar", "Uttar Pradesh", 26.45, 80.33),
    ("Varanasi", "Uttar Pradesh", 25.32, 83.01),
    ("Agra", "Uttar Pradesh", 27.18, 78.02),
    ("Prayagraj", "Uttar Pradesh", 25.44, 81.85),
    ("Ghaziabad", "Uttar Pradesh", 28.67, 77.44),
    ("Meerut", "Uttar Pradesh", 28.98, 77.71),
    ("Bareilly", "Uttar Pradesh", 28.35, 79.42),
    ("Gorakhpur", "Uttar Pradesh", 26.76, 83.37),
    ("Gautam Buddh Nagar", "Uttar Pradesh", 28.54, 77.39),
    ("Patna", "Bihar", 25.59, 85.14),
    ("Gaya", "Bihar", 24.80, 85.00),
    ("Bhagalpur", "Bihar", 25.24, 86.98),
    ("Muzaffarpur", "Bihar", 26.12, 85.39),
    ("Darbhanga", "Bihar", 26.15, 85.90),
    ("Kolkata", "West Bengal", 22.57, 88.36),
    ("Howrah", "West Bengal", 22.59, 88.31),
    ("Darjeeling", "West Bengal", 27.04, 88.26),
    ("Malda", "West Bengal", 25.00, 88.14),
    ("Paschim Bardhaman", "West Bengal", 23.68, 86.97),
    ("Paschim Medinipur", "West Bengal", 22.33, 87.32),
    ("Jaipur", "Rajasthan", 26.91, 75.79),
    ("Jodhpur", "Rajasthan", 26.29, 73.02),
    ("Udaipur", "Rajasthan", 24.58, 73.68),
    ("Kota", "Rajasthan", 25.21, 75.86),
    ("Bikaner", "Rajasthan", 28.02, 73.31),
    ("Ajmer", "Rajasthan", 26.45, 74.64),
    ("Alwar", "Rajasthan", 27.57, 76.60),
    ("Ahmedabad", "Gujarat", 23.02, 72.57),
    ("Surat", "Gujarat", 21.17, 72.83),
    ("Vadodara", "Gujarat", 22.31, 73.18),
    ("Rajkot", "Gujarat", 22.30, 70.80),
    ("Bhavnagar", "Gujarat", 21.76, 72.15),
    ("Jamnagar", "Gujarat", 22.47, 70.07),
    ("Kutch", "Gujarat", 23.25, 69.67),
    ("Bhopal", "Madhya Pradesh", 23.26, 77.41),
    ("Indore", "Madhya Pradesh", 22.72, 75.86),
    ("Jabalpur", "Madhya Pradesh", 23.18, 79.99),
    ("Gwalior", "Madhya Pradesh", 26.22, 78.18),
    ("Ujjain", "Madhya Pradesh", 23.18, 75.78),
    ("Raipur", "Chhattisgarh", 21.25, 81.63),
    ("Bilaspur", "Chhattisgarh", 22.09, 82.14),
    ("Durg", "Chhattisgarh", 21.19, 81.28),
    ("Bastar", "Chhattisgarh", 19.08, 82.03),
    ("Bengaluru Urban", "Karnataka", 12.97, 77.59),
    ("Mysuru", "Karnataka", 12.30, 76.64),
    ("Dakshina Kannada", "Karnataka", 12.87, 74.84),
    ("Belagavi", "Karnataka", 15.85, 74.50),
    ("Dharwad", "Karnataka", 15.36, 75.12),
    ("Kalaburagi", "Karnataka", 17.33, 76.84),
    ("Thiruvananthapuram", "Kerala", 8.52, 76.94),
    ("Ernakulam", "Kerala", 9.98, 76.28),
    ("Kozhikode", "Kerala", 11.26, 75.78),
    ("Thrissur", "Kerala", 10.53, 76.21),
    ("Kannur", "Kerala", 11.87, 75.36),
    ("Idukki", "Kerala", 9.85, 76.97),
    ("Chennai", "Tamil Nadu", 13.08, 80.27),
    ("Coimbatore", "Tamil Nadu", 11.02, 76.96),
    ("Madurai", "Tamil Nadu", 9.93, 78.12),
    ("Tiruchirappalli", "Tamil Nadu", 10.79, 78.70),
    ("Salem", "Tamil Nadu", 11.66, 78.15),
    ("Tirunelveli", "Tamil Nadu", 8.71, 77.76),
    ("The Nilgiris", "Tamil Nadu", 11.41, 76.70),
    ("Visakhapatnam", "Andhra Pradesh", 17.69, 83.22),
    ("NTR (Vijayawada)", "Andhra Pradesh", 16.51, 80.65),
    ("Guntur", "Andhra Pradesh", 16.30, 80.44),
    ("Tirupati", "Andhra Pradesh", 13.63, 79.42),
    ("Kurnool", "Andhra Pradesh", 15.83, 78.04),
    ("Hyderabad", "Telangana", 17.39, 78.49),
    ("Warangal", "Telangana", 17.98, 79.60),
    ("Nizamabad", "Telangana", 18.67, 78.10),
    ("Karimnagar", "Telangana", 18.44, 79.13),
    ("Khordha (Bhubaneswar)", "Odisha", 20.30, 85.82),
    ("Cuttack", "Odisha", 20.46, 85.88),
    ("Puri", "Odisha", 19.81, 85.83),
    ("Sambalpur", "Odisha", 21.47, 83.97),
    ("Balasore", "Odisha", 21.49, 86.93),
    ("Kamrup Metropolitan", "Assam", 26.14, 91.74),
    ("Dibrugarh", "Assam", 27.48, 94.90),
    ("Cachar", "Assam", 24.83, 92.78),
    ("Jorhat", "Assam", 26.75, 94.22),
    ("Amritsar", "Punjab", 31.63, 74.87),
    ("Ludhiana", "Punjab", 30.90, 75.85),
    ("Jalandhar", "Punjab", 31.33, 75.58),
    ("Patiala", "Punjab", 30.34, 76.38),
    ("Bathinda", "Punjab", 30.21, 74.95),
    ("Gurugram", "Haryana", 28.46, 77.03),
    ("Faridabad", "Haryana", 28.41, 77.31),
    ("Panchkula", "Haryana", 30.69, 76.85),
    ("Hisar", "Haryana", 29.15, 75.72),
    ("Karnal", "Haryana", 29.69, 76.99),
    ("Dehradun", "Uttarakhand", 30.32, 78.03),
    ("Haridwar", "Uttarakhand", 29.94, 78.16),
    ("Nainital", "Uttarakhand", 29.38, 79.46),
    ("Uttarkashi", "Uttarakhand", 30.73, 78.44),
    ("Shimla", "Himachal Pradesh", 31.10, 77.17),
    ("Kangra", "Himachal Pradesh", 32.22, 76.32),
    ("Mandi", "Himachal Pradesh", 31.71, 76.93),
    ("Srinagar", "Jammu and Kashmir", 34.08, 74.80),
    ("Jammu", "Jammu and Kashmir", 32.73, 74.87),
    ("Baramulla", "Jammu and Kashmir", 34.20, 74.34),
    ("Anantnag", "Jammu and Kashmir", 33.73, 75.15),
    ("Leh", "Ladakh", 34.16, 77.58),
    ("Ranchi", "Jharkhand", 23.34, 85.31),
    ("East Singhbhum", "Jharkhand", 22.80, 86.19),
    ("Dhanbad", "Jharkhand", 23.80, 86.43),
    ("Bokaro", "Jharkhand", 23.67, 86.15),
    ("North Goa", "Goa", 15.49, 73.83),
    ("South Goa", "Goa", 15.27, 73.96),
    ("Imphal West", "Manipur", 24.82, 93.94),
    ("East Khasi Hills", "Meghalaya", 25.57, 91.88),
    ("Aizawl", "Mizoram", 23.73, 92.72),
    ("Kohima", "Nagaland", 25.67, 94.11),
    ("West Tripura", "Tripura", 23.83, 91.28),
    ("Papum Pare", "Arunachal Pradesh", 27.10, 93.62),
    ("East Sikkim", "Sikkim", 27.33, 88.61),
    ("Puducherry", "Puducherry", 11.94, 79.83),
    ("South Andaman", "Andaman and Nicobar Islands", 11.62, 92.72),
    ("Chandigarh", "Chandigarh", 30.73, 76.78),
    ("Lakshadweep", "Lakshadweep", 10.57, 72.64),
    ("Daman", "Dadra and Nagar Haveli and Daman and Diu", 20.42, 72.83),
]

_LATS = np.array([d[2] for d in DISTRICTS])
_LONS = np.array([d[3] for d in DISTRICTS])


def _km_per_deg(lat):
    return 111.0, 111.0 * np.cos(np.radians(lat))


def nearest_district(lat, lon):
    """Nearest district centroid to a single (lat, lon) point."""
    km_lat, km_lon = _km_per_deg(lat)
    d_km = np.hypot((lat - _LATS) * km_lat, (lon - _LONS) * km_lon)
    idx = int(np.argmin(d_km))
    name, state, _, _ = DISTRICTS[idx]
    return {"district": name, "state": state, "distance_km": round(float(d_km[idx]), 1)}


def nearest_districts_vectorized(lats: np.ndarray, lons: np.ndarray):
    """Same as nearest_district but for an array of points at once — used
    by hazard_india.detect(), which can produce thousands of hail cells
    across the all-India grid; a Python-level loop per cell over ~130
    centroids would be the same kind of avoidable overhead the lightning
    proximity check used to have before it was vectorized (see git log).

    Returns a list of {district, state} dicts, one per input point,
    parallel to `lats`/`lons`.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    km_lat = 111.0
    km_lon_per_point = 111.0 * np.cos(np.radians(lats))  # (N,)

    dlat = (lats[:, None] - _LATS[None, :]) * km_lat  # (N, D)
    dlon = (lons[:, None] - _LONS[None, :]) * km_lon_per_point[:, None]  # (N, D)
    d_km = np.hypot(dlat, dlon)
    idxs = np.argmin(d_km, axis=1)
    return [{"district": DISTRICTS[i][0], "state": DISTRICTS[i][1]} for i in idxs.tolist()]
