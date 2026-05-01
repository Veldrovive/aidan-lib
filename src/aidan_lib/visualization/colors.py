import random
from typing import List, Tuple, Union, Optional

DEFAULT_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Chartreuse
    (0, 128, 255),    # Azure
]

def get_colors(num_colors: int, supplied_colors: Optional[List[Union[Tuple[int, int, int], str]]] = None) -> List[Union[Tuple[int, int, int], str]]:
    """
    Returns a list of colors of length num_colors.
    Uses supplied_colors first, then defaults, then random colors.
    """
    colors: List[Union[Tuple[int, int, int], str]] = []
    if supplied_colors is not None:
        colors.extend(supplied_colors)
    
    # Fill remaining with defaults
    if len(colors) < num_colors:
        needed = num_colors - len(colors)
        colors.extend(DEFAULT_COLORS[:needed])
        
    # If still need more, generate random
    while len(colors) < num_colors:
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
    return colors[:num_colors]
