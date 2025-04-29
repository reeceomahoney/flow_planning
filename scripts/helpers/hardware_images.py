import glob
import os

from PIL import Image


def tile_images(input_folder, output_path, num_rows=2):
    """
    Tile images from a folder into a grid with a specified number of rows.

    Args:
        input_folder: Path to folder containing images
        output_path: Path to save the tiled image
        num_rows: Number of rows in the grid
    """
    # Get all empty_*.png and box_*.png images
    empty_images = sorted(glob.glob(os.path.join(input_folder, "empty_*.png")))
    box_images = sorted(glob.glob(os.path.join(input_folder, "box_*.png")))

    # Combine all images
    all_images = empty_images + box_images

    if not all_images:
        print(f"No images found in {input_folder}")
        return

    # Calculate grid dimensions
    num_images = len(all_images)
    num_cols = (num_images + num_rows - 1) // num_rows  # Ceiling division

    # Load the first image to get dimensions
    first_img = Image.open(all_images[0])
    img_width, img_height = first_img.size

    # Create a new image to hold the grid
    grid_width = num_cols * img_width
    grid_height = num_rows * img_height
    grid_img = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 0))

    # Place each image in the grid
    for i, img_path in enumerate(all_images):
        img = Image.open(img_path)
        row = i // num_cols
        col = i % num_cols
        x = col * img_width
        y = row * img_height
        grid_img.paste(img, (x, y))
        print(f"Placed {os.path.basename(img_path)} at position ({row}, {col})")

    # Save the result
    # grid_img.show()
    grid_img.save(output_path)
    print(f"Created tiled image at {output_path}")


if __name__ == "__main__":
    # Path to your images folder
    input_folder = "data/images/hardware/"

    # Path to save the output image
    output_path = "tiled_hardware_images.png"

    # Create tiled image with 2 rows
    tile_images(input_folder, output_path, num_rows=2)
