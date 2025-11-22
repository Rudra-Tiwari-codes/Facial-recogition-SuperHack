import urllib.request
import os

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    print(f"Downloading {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {output_path}: {e}")
        return False

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Age detection model files
age_prototxt_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
age_prototxt_path = "models/age_deploy.prototxt"

# Download age prototxt
if not os.path.exists(age_prototxt_path):
    download_file(age_prototxt_url, age_prototxt_path)
else:
    print(f"✓ {age_prototxt_path} already exists")

# Check if caffemodel exists
age_model_path = "models/age_net.caffemodel"
if os.path.exists(age_model_path):
    print(f"✓ {age_model_path} already exists")
else:
    print(f"⚠ {age_model_path} not found. You may need to download it manually.")

print("\nModel download complete!")
print("Ready to run the facial recognition system.")
