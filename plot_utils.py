from matplotlib.patches import Rectangle
import numpy as np


def draw_rect_on_plot(ax, objects_rect, linewidth=2, edgecolor='red', facecolor='none'):
    for x0, y0, x1, y1 in objects_rect:
        rect = Rectangle(
            (x0, y0), 
            x1-x0, 
            y1-y0,
            linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor
        )
        ax.add_patch(rect)


def draw_box_edges_on_plot(ax, boxes: np.ndarray, color: str = "red"):
        """
        Draws the 12 edges of a projected 3D bounding box on a Matplotlib Axes.
        """
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7)   # sides
        ]
        for box in boxes:
            for i, j in edges:
                ax.plot(
                    [box[i,0], box[j,0]],
                    [box[i,1], box[j,1]],
                    color=color, linewidth=1.5
                )


def draw_points_on_plot(ax, img_pts, depth, plt_size: tuple):
    mask = (
            (depth > 0) &
            (img_pts[:,0] >= 0) & (img_pts[:,1] >= 0) &
            (img_pts[:,0] < plt_size[1]) & (img_pts[:,1] < plt_size[0])
        )
    pts, dp = img_pts[mask], depth[mask]
    ax.scatter(pts[:,0], pts[:,1], c=dp, cmap='jet', s=1, alpha=0.3)

