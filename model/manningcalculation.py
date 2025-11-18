import requests

def fetch_river_data(https://mausam.imd.gov.in/api/districtwise_rainfall_api.php):
    """
    Fetch river parameters from an API.
    
    Expected API response:
    {
        "n": 0.035,                   # Manning's roughness coefficient
        "area": 50,                   # Cross-sectional area (m^2)
        "wetted_perimeter": 20,       # Wetted perimeter (m)
        "slope": 0.001                # Slope (m/m)
    }
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

def manning_velocity(n, area, wetted_perimeter, slope):
    """
    Calculate flow velocity using Manning's equation.
    """
    R = area / wetted_perimeter  # Hydraulic radius
    V = (1 / n) * (R ** (2/3)) * (slope ** 0.5)
    return V

# Example API endpoint (replace with real one)
api_url = "https://example.com/river_data"

try:
    river_data = fetch_river_data(api_url)
    
    velocity = manning_velocity(
        n=river_data["n"],
        area=river_data["area"],
        wetted_perimeter=river_data["wetted_perimeter"],
        slope=river_data["slope"]
    )
    
    print(f"Predicted Flow Velocity: {velocity:.2f} m/s")

except Exception as e:
    print(e)
