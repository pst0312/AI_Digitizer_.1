import os
from pathlib import Path
import cv2
from doclayout_yolo import YOLOv10

def setup_pipeline():
    """Ensure folders and model paths are correctly established."""
    current_dir = Path(__file__).resolve().parent
    
    # 1. Point directly to your local downloaded model file
    model_filename = "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_path = current_dir / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find '{model_filename}' in {current_dir}. "
            f"Please download it from Hugging Face and place it here."
        )
        
    # 2. Setup input and output directories
    input_folder = current_dir / "input_pages"
    output_folder = current_dir / "output_crops"
    
    input_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)
    
    return model_path, input_folder, output_folder

def extract_graphs():
    model_path, input_folder, output_folder = setup_pipeline()
    
    print("Loading local DocLayout-YOLO model...")
    # Initialize the model using your local path
    model = YOLOv10(str(model_path))
    
    # Supported image extensions
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff"}
    image_paths = [p for p in input_folder.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not image_paths:
        print(f"\n[!] No images found in '{input_folder.name}' directory. Add page images to process.")
        return

    print(f"Found {len(image_paths)} images to analyze.\n")

    for img_path in image_paths:
        print(f"Analyzing {img_path.name}...")
        
        # Run prediction locally. Use device="cuda:0" if you have an NVIDIA GPU
        results = model.predict(str(img_path), imgsz=1024, conf=0.25, device="cpu")
        
        # Load image via OpenCV for cropping
        image = cv2.imread(str(img_path))
        height, width, _ = image.shape
        
        # Make a specific output folder for this document page
        page_output_dir = output_folder / img_path.stem
        page_output_dir.mkdir(exist_ok=True)
        
        crop_count = 0

        # Loop through detected boundaries
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Filter specifically for document regions likely containing graphs
            if label in ["figure", "picture", "table"]:
                crop_count += 1
                
                # Extract raw pixel coordinates
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                
                # Apply a 5% padding to avoid clipping labels or axis numbers
                pad_x = int((xmax - xmin) * 0.05)
                pad_y = int((ymax - ymin) * 0.05)
                
                xmin = max(0, xmin - pad_x)
                ymin = max(0, ymin - pad_y)
                xmax = min(width, xmax + pad_x)
                ymax = min(height, ymax + pad_y)
                
                # Perform the pixel slice
                cropped_segment = image[ymin:ymax, xmin:xmax]
                
                # Save the isolated asset
                output_file_name = f"{label}_crop_{crop_count}.png"
                output_file_path = page_output_dir / output_file_name
                
                cv2.imwrite(str(output_file_path), cropped_segment)
                print(f"  -> Extracted {label} to {output_file_path.relative_to(output_folder.parent)}")
                
        if crop_count == 0:
            print("  -> No graphs or figures detected on this page.")

if __name__ == "__main__":
    extract_graphs()