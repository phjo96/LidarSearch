import cv2
import numpy as np
import argparse


def main(input: str, output: str) -> None:
    """Detect lines in a lidar image

    Args:
        input (str): path to image
        output (str): destination path
    """

    image = cv2.imread(input, -1)

    # Intermediate image for storing lines
    line_image = np.zeros_like(image)

    # Filter for removing noise
    kernel = np.ones((25, 25))

    # Line finder parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 400  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 800  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    # Slicing variables
    mean = np.mean(image[image > 0])
    std = np.std(image[image > 0])
    step = 4
    overlap = 1

    # Slice up the image by depth
    for i, slice in enumerate(range(int(mean - std), int(mean + 1.5 * std), step)):
        temp = np.zeros_like(image)
        temp[image > slice] = 1
        temp[image > slice + step + overlap] = 0

        # Filter out points with to many/few neighbors
        filtered = cv2.filter2D(temp, -1, kernel)
        temp[filtered > 200] = 0
        temp[filtered < 10] = 0

        # Get lines
        lines = cv2.HoughLinesP(
            temp, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
        )

        # Paint the lines
        if lines is not None and len(lines) < 40:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)

    # Combine the lines
    min_line_length = 1000
    lines = cv2.HoughLinesP(
        line_image, rho, theta, threshold, np.array([]), min_line_length, 300
    )

    # Display
    for line in lines:
        for x1, y1, x2, y2 in line:
            print(x1, y1, x2, y2)
            cv2.line(image, (x1, y1), (x2, y2), 255, 3)

    cv2.imwrite(output, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="data/LiDAR.jpg")
    parser.add_argument("--output", "-o", type=str, default="data/lines.jpg")

    args = parser.parse_args()
    main(args.input, args.output)
