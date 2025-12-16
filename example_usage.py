"""
Example usage script for PigeonPhoto_Automator.
Demonstrates programmatic usage of the pipeline.
"""

from pathlib import Path
from src.pipeline import PigeonPipeline
import src.config as config

def main():
    """Example usage of the pipeline."""
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = PigeonPipeline()
    print("Pipeline initialized!")
    
    # Example file paths (update these to your actual image paths)
    body_image_path = "path/to/Body_Raw.jpg"
    eye_image_path = "path/to/Eye_Macro.jpg"
    
    # Check if files exist
    if not Path(body_image_path).exists():
        print(f"Error: Body image not found at {body_image_path}")
        print("Please update the paths in this script with your actual image paths.")
        return
    
    if not Path(eye_image_path).exists():
        print(f"Error: Eye image not found at {eye_image_path}")
        print("Please update the paths in this script with your actual image paths.")
        return
    
    # Process images
    print("Processing images...")
    try:
        result = pipeline.process(
            body_image_path=body_image_path,
            eye_image_path=eye_image_path,
            name="Champion",
            ring_number="2024-001"
        )
        
        print("Processing complete!")
        
        # Save output
        output_path = "output/golden_prince_example.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_output(result, output_path)
        
        print(f"Output saved to {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

