import os
import requests
from datetime import datetime, timedelta
from pystac_client import Client
import numpy as np
import json
import pickle
import torch
import random
import logging
from typing import Union, List, Tuple, Dict, Any

# Constants for configuration
CACHE_DIR = "data_cache"
SENTINEL_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/stac/"
COPERNICUS_API_URL = "https://services.sentinel-hub.com/api/v1/process"
MET_MALAYSIA_API_URL = "https://api.met.gov.my/v2.1/data"
FLOOD_API_URL = "https://api.open-meteo.com/v1/flood"
TRAIN_SPLIT = 0.8

# Set up logging
logging.basicConfig(level=logging.INFO)


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize numpy array data to range [0, 1]."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class FloodDataCollector:
    def __init__(self, geojson_path_or_dict: Union[str, Dict[str, Any]], start_date: datetime, end_date: datetime,
                 resolution: timedelta, output_dir: Union[str, os.PathLike]):
        self.output_dir = output_dir
        self.geojson_data = self._load_geojson(geojson_path_or_dict)
        self.start_date = start_date
        self.end_date = end_date
        self.resolution = resolution
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def _load_geojson(self, geojson_path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load GeoJSON data from file or dictionary."""
        if isinstance(geojson_path_or_dict, (str, os.PathLike)):
            with open(geojson_path_or_dict) as f:
                return json.load(f)
        return geojson_path_or_dict

    def _get_sentinel1_data(self, bbox: List[float], time_range: List[str]) -> np.ndarray:
        """Retrieve Sentinel-1 data for the specified bounding box and time range."""
        cache_key = f"sentinel1_{time_range[0]}_{time_range[1]}.pkl"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        catalog = Client.open(SENTINEL_CATALOG_URL)
        search = catalog.search(
            collections=["SENTINEL-1"],
            bbox=bbox,
            datetime=time_range,
            query={"sar:instrument_mode": "IW", "polarization": "VV VH"},
        )
        data = []
        for item in search.get_items():
            vv_data = self._fetch_asset_data(item, "amplitude_vv")
            vh_data = self._fetch_asset_data(item, "amplitude_vh")
            data.append((vv_data, vh_data))

        if data:
            data = np.array(data)
            data = normalize_data(data)
        else:
            data = np.array([])

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        return data

    def _fetch_asset_data(self, item: Any, asset_key: str) -> np.ndarray:
        """Fetch asset data from the provided item."""
        href = item.assets[asset_key].href
        response = requests.get(href)
        return np.frombuffer(response.content, dtype=np.float32)

    def _get_copernicus_dem(self, bbox: List[float]) -> np.ndarray:
        """Retrieve Copernicus DEM data for the specified bounding box."""
        cache_key = "copernicus_dem.pkl"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        request_data = {
            "input": {"bounds": {"bbox": bbox}, "data": [{"type": "dem"}]},
            "output": {"width": 512, "height": 512},
        }
        response = requests.post(COPERNICUS_API_URL, json=request_data)
        data = response.json()

        elevation = np.array(data["elevation"])
        slope = np.gradient(elevation)
        elevation = normalize_data(elevation)
        slope = normalize_data(slope)
        data = np.stack((elevation, slope), axis=-1)

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        return data

    def _get_met_malaysia_data(self, location_id: str, time_range: List[str]) -> np.ndarray:
        """Retrieve meteorological data from Met Malaysia API for the specified location and time range."""
        cache_key = f"met_malaysia_{location_id}_{time_range[0]}_{time_range[1]}.pkl"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        url = f"{MET_MALAYSIA_API_URL}?datasetid=FORECAST&datacategoryid=GENERAL&locationid={location_id}&start_date={time_range[0]}&end_date={time_range[1]}"
        response = requests.get(url)
        json_data = response.json()

        precipitation, temperature = self._extract_met_data(json_data)
        if precipitation.size > 0 and temperature.size > 0:
            precipitation = normalize_data(precipitation)
            temperature = normalize_data(temperature)
            data = np.stack((precipitation, temperature), axis=-1)
        else:
            data = np.array([])

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        return data

    def _extract_met_data(self, json_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract precipitation and temperature data from the Met Malaysia API response."""
        precipitation = []
        temperature = []
        for item in json_data.get("metadata", []):
            if item["name"] == "Precipitation":
                precipitation = item["data"]
            elif item["name"] == "Temperature":
                temperature = item["data"]
        return np.array(precipitation), np.array(temperature)

    def _get_flood_data(self, latitude: float, longitude: float, time_range: List[str]) -> List[int]:
        """Retrieve flood data from the Open-Meteo API for the specified location and time range."""
        url = f"{FLOOD_API_URL}?latitude={latitude}&longitude={longitude}&start_date={time_range[0]}&end_date={time_range[1]}"
        response = requests.get(url)
        json_data = response.json()
        discharge_data = json_data.get("discharge", [])
        return [int(discharge > 1000) for discharge in discharge_data]

    def collect_data(self):
        """Collect and process flood-related data."""
        bbox = self.geojson_data["features"][0]["geometry"]["coordinates"][0]
        bbox = [min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1]), max(bbox[2][0], bbox[3][0]),
                max(bbox[2][1], bbox[3][1])]
        latitude = (bbox[1] + bbox[3]) / 2
        longitude = (bbox[0] + bbox[2]) / 2

        # TODO: Get the appropriate location_id for the given latitude and longitude
        location_id = "TODO"

        current_date = self.start_date
        data = []
        labels = []
        while current_date <= self.end_date:
            time_range = [current_date.strftime("%Y-%m-%d"), (current_date + self.resolution).strftime("%Y-%m-%d")]
            logging.info(f"Processing time range: {time_range}")

            sentinel1_data = self._get_sentinel1_data(bbox, time_range)
            dem_data = self._get_copernicus_dem(bbox)
            met_data = self._get_met_malaysia_data(location_id, time_range)
            flood_data = self._get_flood_data(latitude, longitude, time_range)

            if sentinel1_data.size > 0 and dem_data.size > 0 and met_data.size > 0:
                combined_data = np.concatenate((sentinel1_data, dem_data, met_data), axis=-1)
                data.append(combined_data)
                labels.append(flood_data)
            else:
                logging.warning(f"Skipping time range {time_range} due to missing data.")

            current_date += self.resolution

        # Balance the dataset
        data, labels = self._balance_dataset(data, labels)

        # Split the dataset into train and validation sets
        train_data, val_data, train_labels, val_labels = self._split_dataset(data, labels)

        # Save the train and validation datasets
        torch.save({"data": train_data, "labels": train_labels}, os.path.join(self.output_dir, "train_dataset.pt"))
        torch.save({"data": val_data, "labels": val_labels}, os.path.join(self.output_dir, "val_dataset.pt"))

    def _balance_dataset(self, data: List[np.ndarray], labels: List[List[int]]) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Balance the dataset by oversampling the minority class."""
        flood_indices = [i for i, label in enumerate(labels) if any(label)]
        no_flood_indices = [i for i, label in enumerate(labels) if not any(label)]

        num_samples = max(len(flood_indices), len(no_flood_indices))
        balanced_data = []
        balanced_labels = []

        for _ in range(num_samples):
            if len(flood_indices) > 0:
                index = random.choice(flood_indices)
                balanced_data.append(data[index])
                balanced_labels.append(labels[index])
                flood_indices.remove(index)
            if len(no_flood_indices) > 0:
                index = random.choice(no_flood_indices)
                balanced_data.append(data[index])
                balanced_labels.append(labels[index])
                no_flood_indices.remove(index)

        return balanced_data, balanced_labels

    def _split_dataset(self, data: List[np.ndarray], labels: List[List[int]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], List[List[int]]]:
        """Split the dataset into train and validation sets."""
        num_samples = len(data)
        num_train = int(num_samples * TRAIN_SPLIT)

        train_data = data[:num_train]
        val_data = data[num_train:]
        train_labels = labels[:num_train]
        val_labels = labels[num_train:]

        return train_data, val_data, train_labels, val_labels


if __name__ == "__main__":
    geojson_path = "path/to/your/geojson/file.geojson"
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 1, 31)
    resolution = timedelta(hours=1)
    output_dir = "output"

    collector = FloodDataCollector(geojson_path, start_date, end_date, resolution, output_dir)
    collector.collect_data()