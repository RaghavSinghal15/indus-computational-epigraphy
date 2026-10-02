import os
import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt

def augment_image(img):
    """
    Apply random distortions to an image to imitate worn carvings.
    Assuming img is grayscale, white text on black background.
    """
    h, w = img.shape
    
    # 1. Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Using borderValue=0 (black) to match background
    img_aug = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    # 2. Random stroke thickness (morphology)
    kernel_size = random.choice([2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickness_choice = random.random()
    if thickness_choice > 0.6:
        # Thicker strokes (dilate)
        img_aug = cv2.dilate(img_aug, kernel, iterations=1)
    elif thickness_choice < 0.4:
        # Thinner strokes (erode)
        img_aug = cv2.erode(img_aug, kernel, iterations=1)
        
    # 3. Random Blur
    if random.random() > 0.3:
        blur_amount = random.choice([3, 5])
        img_aug = cv2.GaussianBlur(img_aug, (blur_amount, blur_amount), 0)
        
    # 4. Noise
    if random.random() > 0.3:
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, img_aug.shape).astype(np.float32)
        img_aug = np.clip(img_aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return img_aug

def main():
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "data" / "raw" / "sign_templates"
    synthetic_dir = repo_root / "data" / "processed" / "synthetic"
    figures_dir = repo_root / "reports" / "figures"
    
    # Ensure output directories exist
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # List all template files
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    if not template_files:
        print(f"No templates found in {templates_dir}")
        return
        
    np.random.seed(42)
    random.seed(42)
    
    COPIES_PER_SIGN = 50
    
    # Sample some random templates for the grid
    example_templates = random.sample(template_files, min(5, len(template_files)))
    grid_examples = {t.stem: [] for t in example_templates}
    
    print(f"Generating {COPIES_PER_SIGN} synthetic copies for each of {len(template_files)} templates...")
    
    for i, t_path in enumerate(template_files):
        sign_id = t_path.stem
        out_folder = synthetic_dir / sign_id
        out_folder.mkdir(exist_ok=True)
        
        img = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {t_path}")
            continue
            
        for c in range(COPIES_PER_SIGN):
            img_aug = augment_image(img)
            out_name = f"{sign_id}_syn_{c:02d}.png"
            cv2.imwrite(str(out_folder / out_name), img_aug)
            
            # Save for grid if this is an example template
            if sign_id in grid_examples and len(grid_examples[sign_id]) < 5:
                grid_examples[sign_id].append(img_aug)
                
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(template_files)} templates...")
            
    print("Generation complete.")
    
    # Generate example grid
    if grid_examples:
        fig, axes = plt.subplots(len(example_templates), 6, figsize=(12, len(example_templates)*2))
        
        for row_idx, t_path in enumerate(example_templates):
            sign_id = t_path.stem
            orig_img = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
            
            # Plot original in the first column
            ax = axes[row_idx, 0] if len(example_templates) > 1 else axes[0]
            ax.imshow(orig_img, cmap='gray')
            ax.set_title(f"Orig: {sign_id}")
            ax.axis('off')
            
            # Plot 5 synthetics in the other columns
            syn_imgs = grid_examples[sign_id]
            for col_idx, s_img in enumerate(syn_imgs):
                ax = axes[row_idx, col_idx + 1] if len(example_templates) > 1 else axes[col_idx + 1]
                ax.imshow(s_img, cmap='gray')
                ax.set_title(f"Syn {col_idx+1}")
                ax.axis('off')
                
        plt.tight_layout()
        grid_path = figures_dir / "synthetic_examples.png"
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"Saved example grid to {grid_path}")

if __name__ == "__main__":
    main()
