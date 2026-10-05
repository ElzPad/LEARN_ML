import math

def rotation_layer(X, angle):
    """
    Rotate 2D points by `angle` radians and return the rotated [x, y] pairs.

    Args:
        X (list[list[float]]): A list of 2D points, where each point is represented as [x, y]
        angle (float): Rotation angle in radians. Positive angles rotate points counterclockwise

    Returns:
        list[list[float]]: The rotated points, each represented as [x, y]
    """
    rotation_matrix = [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ]

    return [
        [
            x * rotation_matrix[0][0] + y * rotation_matrix[0][1],
            x * rotation_matrix[1][0] + y * rotation_matrix[1][1]
        ]
        for x, y in X
    ]

if __name__ == "__main__":
    result = [[round(v, 4) for v in row] for row in rotation_layer([[1, 0], [0, 1]], 0.5)]
    print(result) # Expected output: [[0.8776, 0.4794], [-0.4794, 0.8776]]