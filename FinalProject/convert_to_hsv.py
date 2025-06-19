# convert images in a directory to HSV format
import os
from PIL import Image

def convert_images_to_hsv(directory: str, output_directory: str):
    """
    Convert all images in a directory to HSV format and save them to an output directory.
    
    :param directory: Path to the input directory containing images.
    :param output_directory: Path to the output directory where converted images will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            hsv_img = img.convert('HSV')
            # save as HSV format not JPEG
            hsv_img.save(os.path.join(output_directory, filename), format='PNG')
            
def main():
    input_directory = './Fruits Classification/all'  # Replace with your input directory
    output_directory = './Fruits Classification/all/hsv'  # Replace with your desired output directory
    convert_images_to_hsv(input_directory, output_directory)
    print(f"Converted images saved to {output_directory}")
    
if __name__ == "__main__":
    main()