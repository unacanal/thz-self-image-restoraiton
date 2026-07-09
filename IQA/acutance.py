import sys
import cv2
import numpy as np

def compute_vol(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    vol = np.var(laplacian)

    return vol

def compute_acutance(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Compute the gradient in the X and Y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Compute the acutance as the mean of the gradient magnitudes
    acutance = np.mean(magnitude)

    return acutance

def compute_canny_edge_sharpness(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Canny 엣지 디텍터를 사용하여 엣지를 찾기
    edges = cv2.Canny(image,100,200)

    # 엣지 픽셀의 비율을 계산하여 샤프니스 평가
    sharpness = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

    return sharpness

# Test the function with the path to your image file
acutance = compute_acutance(sys.argv[1])
vol = compute_vol(sys.argv[1])
print(f"Acutance: {acutance}, Vol: {vol}, Canny: {compute_canny_edge_sharpness(sys.argv[1])}")
