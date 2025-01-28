import csv
import os
import pyvips
import openslide
from openslide import OpenSlide

# Configure paths
csv_file = "notebooks/cropping_coordinates.csv"
image_dir = "/scratch/project_2003009/space_he/adjacent_slide/"
output_dir = "/scratch/project_2003009/space_he/adjacent_slide_cropped/"
os.makedirs(output_dir, exist_ok=True)


def get_microns_per_pixel(image_path):
    """Extract microns per pixel from the image metadata."""
    try:
        slide = OpenSlide(image_path)
        if 'openslide.mpp-x' in slide.properties:
            mpp_x = float(slide.properties['openslide.mpp-x'])
            mpp_y = float(slide.properties['openslide.mpp-y'])
            print(f"Microns per Pixel: X = {mpp_x}, Y = {mpp_y}")
            return mpp_x, mpp_y
        else:
            # Fallback to VIPS if OpenSlide metadata isn't available
            image = pyvips.Image.new_from_file(image_path, access="sequential")
            x_res = image.get("XResolution") if image.get_typeof("XResolution") else None
            y_res = image.get("YResolution") if image.get_typeof("YResolution") else None

            if x_res and y_res:
                microns_per_pixel_x = 10000 / x_res
                microns_per_pixel_y = 10000 / y_res
                return microns_per_pixel_x, microns_per_pixel_y

        return None, None
    except Exception as e:
        print(f"Error extracting microns per pixel: {e}")
        return None, None


def crop_resize_and_save_pyramidal_tiff(image_path, x, y, width, height, output_file, microns_per_pixel):
    try:
        # Load the image using VIPS
        slide = pyvips.Image.new_from_file(image_path, access="sequential")

        if slide.bands < 3:
            slide = slide.colourspace("rgb")

        # Crop the region
        cropped_region = slide.crop(x, y, width, height)

        # Calculate new dimensions (25% of original)
        new_width = int(width * 0.25)
        new_height = int(height * 0.25)

        # Resize the image to 25% of original size
        # Using 'cubic' interpolation for better quality
        resized_region = cropped_region.resize(0.25, kernel="cubic")

        # Adjust the DPI to reflect the new resolution
        # Since we're reducing resolution by 4x, we need to multiply microns_per_pixel by 4
        new_microns_per_pixel = microns_per_pixel * 4
        pixels_per_cm = 10000 / new_microns_per_pixel

        # Ensure the output file has .tiff extension
        output_file = output_file.replace('.svs', '.tiff')

        # Save as pyramidal TIFF with basic properties
        resized_region.tiffsave(
            output_file,
            compression="jpeg",
            Q=90,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            bigtiff=True,
            predictor="horizontal",
            xres=4000.0,
            yres=4000.0,
            resunit='cm',
            properties=True
        )

        print(f"Successfully saved resized cropped region as pyramidal TIFF: {output_file}")
        print(f"Original size: {width}x{height}, New size: {new_width}x{new_height}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Read the CSV and process each entry
with open(csv_file, mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            filename = row["Filename"]
            x = int(row["X"])
            y = int(row["Y"])
            width = int(row["Width"])
            height = int(row["Height"])

            image_path = os.path.join(image_dir, filename)
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}, skipping...")
                continue

            # Use .tiff extension
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.tiff")

            microns_x, microns_y = get_microns_per_pixel(image_path)
            if microns_x and microns_y:
                crop_resize_and_save_pyramidal_tiff(image_path, x, y, width, height, output_file, microns_x)
            else:
                print("Using default microns per pixel value.")
                crop_resize_and_save_pyramidal_tiff(image_path, x, y, width, height, output_file,
                                                    microns_per_pixel=0.25)  # 40x default

        except Exception as e:
            print(f"Error processing row: {row}, Error: {e}")

print("Cropping and resizing process complete.")