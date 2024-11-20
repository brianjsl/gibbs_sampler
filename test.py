def is_bordering_different_value(grid, row, col):
    """
    Checks whether the pixel at (row, col) is bordering a pixel of a different value.
    
    Parameters:
        grid (np.ndarray): 2D array representing the grid.
        row (int): Row index of the pixel.
        col (int): Column index of the pixel.
    
    Returns:
        bool: True if the pixel is bordering a pixel of a different value, False otherwise.
    """
    current_value = grid[row, col]
    rows, cols = grid.shape

    # Define neighbor positions (up, down, left, right)
    neighbors = [
        (row - 1, col),  # Above
        (row + 1, col),  # Below
        (row, col - 1),  # Left
        (row, col + 1)   # Right
    ]
    
    for r, c in neighbors:
        # Check if the neighbor is within bounds
        if 0 <= r < rows and 0 <= c < cols:
            # Check if the neighbor has a different value
            if grid[r, c] != current_value:
                return True
    
    return False


def fraction_bordering_different_value(grid):
    """
    Calculates the fraction of pixels in the grid that border a pixel with a different value.
    
    Parameters:
        grid (np.ndarray): 2D array representing the grid.
    
    Returns:
        float: Fraction of pixels that border a pixel with a different value.
    """
    rows, cols = grid.shape
    bordering_count = 0
    total_pixels = rows * cols

    # Loop through all pixels in the grid
    for row in range(rows):
        for col in range(cols):
            if is_bordering_different_value(grid, row, col):
                bordering_count += 1
    
    # Calculate fraction
    fraction = bordering_count / total_pixels
    return fraction