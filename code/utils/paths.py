from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PAN16_PICKLE_DIR = RAW_DATA_DIR / "pan16_raw" / "pickle_format"
WINOGENDER_DATA = RAW_DATA_DIR / "winogender_raw" / "counterfactual_winogender.csv"
