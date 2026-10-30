"""One shared schema prevents training and inference from reordering inputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "NOX"
TRAIN_YEARS = [2011, 2012]
VALIDATION_YEARS = [2013]
TEST_YEARS = [2014, 2015]
EXPECTED_ROWS = {2011: 7411, 2012: 7628, 2013: 7152, 2014: 7158, 2015: 7384}
SOURCE_URL = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
ARCHIVE_SHA256 = "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set"
FEATURE_LABELS = {
    "AT": "Ambient temperature (°C)",
    "AP": "Ambient pressure (mbar)",
    "AH": "Ambient humidity (%)",
    "AFDP": "Air filter pressure difference (mbar)",
    "GTEP": "Turbine exhaust pressure (mbar)",
    "TIT": "Turbine inlet temperature (°C)",
    "TAT": "Turbine after temperature (°C)",
    "TEY": "Turbine energy yield (MWh)",
    "CDP": "Compressor discharge pressure (source units)",
}
