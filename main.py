import cv2
import numpy as np

def pad_to_power_of_two(image):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Find the nearest power of two for both dimensions
    new_height = 2 ** int(np.ceil(np.log2(height)))
    new_width = 2 ** int(np.ceil(np.log2(width)))

    # Calculate the amount of padding needed on each side
    top_pad = (new_height - height) // 2
    bottom_pad = new_height - height - top_pad
    left_pad = (new_width - width) // 2
    right_pad = new_width - width - left_pad

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

    return padded_image

# Example usage
image = cv2.imread('Skeletonization_Input.bmp', cv2.IMREAD_COLOR)  # Replace 'input_image.bmp' with your BMP image file
padded_image = pad_to_power_of_two(image)

# Save padded image as BMP
cv2.imwrite('Skeletonization_Input2.bmp', padded_image)  # Save padded image as 'padded_image.bmp'