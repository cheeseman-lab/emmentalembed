import torch
import esm
import os

def download_model(model_name, output_dir):
    """Download an ESM model and save it directly, if it doesn't already exist"""
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    regression_path = os.path.join(output_dir, f"{model_name}-contact-regression.pt")
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
    else:
        print(f"Downloading {model_name}...")
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        
        try:
            # Download without using hub cache
            response = torch.hub.download_url_to_file(
                url,
                model_path,
                progress=True
            )
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            return
    
    # Handle regression weights if needed
    needs_regression = not ("esm1v" in model_name or "esm_if" in model_name or 
                          "270K" in model_name or "500K" in model_name)
    
    if needs_regression:
        if os.path.exists(regression_path):
            print(f"Regression weights for {model_name} already exist at {regression_path}")
        else:
            print(f"Downloading regression weights for {model_name}...")
            regression_url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
            try:
                torch.hub.download_url_to_file(
                    regression_url,
                    regression_path,
                    progress=True
                )
                print(f"Successfully downloaded regression weights for {model_name}")
            except Exception as e:
                print(f"Error downloading regression weights for {model_name}: {str(e)}")

def main():
    # Define model groups
    model_groups = {
        "15B": ["esm2_t48_15B_UR50D"],
        "3B": ["esm2_t36_3B_UR50D"],
        "650M": ["esm1b_t33_650M_UR50S", "esm1v_t33_650M_UR90S_1", "esm2_t33_650M_UR50D"]
    }
    
    # Set output directory
    base_output_dir = "models"  # You can change this to your preferred directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Download each group of models
    for group_name, models in model_groups.items():
        print(f"\nProcessing {group_name} models...")
        
        for model_name in models:
            download_model(model_name, base_output_dir)

if __name__ == "__main__":
    main()