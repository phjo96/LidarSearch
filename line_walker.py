import cv2
import numpy as np
from typing import Tuple


def line_walker(
    image: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    deviation: int = 2,
    seg_len: int = 25,
    ext_len: int = 20,
):
    """[summary]

    Args:
        image (np.ndarray): LiDAR image
        p1 (Tuple[int, int]): Starting point 1
        p2 (Tuple[int, int]): Starting point 2
        deviation (int, optional): Maximum height deviation from line. Defaults to 2.
        seg_len (int, optional): Length of used line segment. Defaults to 25.
        ext_len (int, optional): Search length for new points. Defaults to 20.
    """

    # Line points
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])

    # Interpolate between the points
    k, m = np.polyfit(x, y, 1)
    inter_x = np.array(range(x[0], x[1] + 1))
    inter_y = (np.round(k * inter_x + m, 0)).astype(int)

    # Keep the points that are within the correct height range
    mean = np.mean(image[y, x])
    height = image[inter_y, inter_x]
    keep = (mean + deviation >= height) * (height >= mean - deviation)
    x = inter_x[keep]
    y = inter_y[keep]

    # Keep a copy for visualization
    im_copy = image.copy()
    for _ in range(30):

        # Sort the points
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Calculate the mean height and polynomial of the points
        mean = np.mean(image[y[-seg_len:], x[-seg_len:]])
        k, m = np.polyfit(x[-seg_len:], y[-seg_len:], 1)

        x_ = np.array([])
        y_ = np.array([])

        # Draw the line outwards and search around it
        for i in [-1, 0, 1]:

            # Interpolate a new line and keep the points that fit the height
            inter_x = np.array(range(x[-1], x[-1] + ext_len))
            inter_y = (np.round(k * inter_x + m + i, 0)).astype(int)
            height = image[inter_y, inter_x]
            keep = (mean + deviation >= height) * (height >= mean - deviation)

            x_ = np.append(x_, inter_x[keep])
            y_ = np.append(y_, inter_y[keep])

        # Stop if the line ends
        if len(x_) == 0:
            break

        # Stop if outside image
        if (
            any(x_ < 0)
            or any(y_ < 0)
            or any(x_ > image.shape[1])
            or any(y_ > image.shape[0])
        ):
            break

        # Add the new line segment
        x = (np.append(x, x_)).astype(int)
        y = (np.append(y, y_)).astype(int)

        # Visualize
        im_copy[y, x] = 255
        cv2.imshow("Lines", im_copy)
        cv2.waitKey(200)

    image[y, x] = 255
    cv2.imshow("Lines", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    p1 = (246, 329)
    p2 = (256, 321)
    image = cv2.imread("data/LiDAR.jpg", -1)
    image = image[4500:5500, 4500:5500]

    line_walker(image, p1, p2)
