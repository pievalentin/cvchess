import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List

class ChessboardAnalyzer:
    @staticmethod
    def plot_grid_on_transformed_image(image: Image.Image) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        corners = np.array([[0,0], 
                            [image.size[0], 0], 
                            [0, image.size[1]], 
                            [image.size[0], image.size[1]]])
        
        from utils import order_points
        corners = order_points(corners)

        plt.figure(figsize=(10, 10), dpi=80)
        plt.imshow(image)
        
        TL, TR, BR, BL = corners

        def interpolate(xy0: Tuple[float, float], xy1: Tuple[float, float]) -> List[Tuple[float, float]]:
            x0, y0 = xy0
            x1, y1 = xy1
            dx = (x1-x0) / 8
            dy = (y1-y0) / 8
            return [(x0+i*dx, y0+i*dy) for i in range(9)]

        ptsT = interpolate(TL, TR)
        ptsL = interpolate(TL, BL)
        ptsR = interpolate(TR, BR)
        ptsB = interpolate(BL, BR)
            
        for a, b in zip(ptsL, ptsR):
            plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
        for a, b in zip(ptsT, ptsB):
            plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
            
        plt.axis('off')
        plt.savefig('chessboard_transformed_with_grid.jpg')
        return ptsT, ptsL