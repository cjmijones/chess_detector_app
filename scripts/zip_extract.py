from utils import extract_zipfile
from dataset import ChessRecognitionDataset
from pathlib import Path
import shutil

raw_data_dir = Path(__file__).resolve().parent.parent / "raw_data"
zip_file = raw_data_dir / "chessred.zip"
extracted_dir = Path(__file__).resolve().parent.parent / "extracted_data"

# Create target directory if it doesn't exist
extracted_dir.mkdir(parents=True, exist_ok=True)
print(f"Target directory at {extracted_dir}")

# Extract to extracted_data
print(f"Extracting zip file from {zip_file}")
extract_zipfile(str(zip_file), str(extracted_dir))

print(f"Copying annotations.json file from raw_data\n")
shutil.copy(raw_data_dir / "annotations.json", extracted_dir / "annotations.json")

print("Loading Chess Recognition Dataset")
dataset = ChessRecognitionDataset(dataroot="extracted_data", split="train")

# Check one sample
img, labels = dataset[0]
print(img.shape)     # e.g., torch.Size([3, H, W])
print(labels.shape)