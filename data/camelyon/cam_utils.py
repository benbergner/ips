import os
import fnmatch
from collections import namedtuple
from typing import Dict

from PIL import Image
from PIL import ImageDraw
from progress.bar import IncrementalBar

Point = namedtuple('Point', 'x y')

def find_files(pattern, path) -> Dict[str, str]:
    """
    Find files in a directory by given file name pattern.

    Parameters
    ----------
    pattern : str
        File pattern allowing wildcards.

    path : str
        Root directory to search in.

    Returns
    -------
    dict(str: str)
        Dictionary of all found files where the file names are keys and the relative paths
        from search root are values.
    """
    result = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[name] = os.path.join(root, name)
    return result

class ProgressBar(IncrementalBar):
    @property
    def remaining_fmt(self):
        m, s = divmod(self.eta, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'

    @property
    def elapsed_fmt(self):
        m, s = divmod(self.elapsed, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'

def draw_polygon(image: Image.Image, polygon, *, fill, outline) -> Image.Image:
    """
    Draw a filled polygon on to an image.

    Parameters
    ----------
    image : Image.Image
        Background image to be drawn on.

    polygon :
        Polygon to be drawn.

    fill : color str or tuple
        Fill color.

    outline : color str or tuple
        Outline color.

    Returns
    -------
    Image.Image
        A copy of the background image with the polygon drawn onto.
    """
    img_back = image
    img_poly = Image.new('RGBA', img_back.size)
    img_draw = ImageDraw.Draw(img_poly)
    img_draw.polygon(polygon, fill, outline)
    img_back.paste(img_poly, mask=img_poly)
    return img_back

def get_relative_polygon(polygon, origin: Point, downsample=1):
    """
    Translate the polygon to relative to a point.

    Parameters
    ----------
    polygon : Sequence[Point]
        Polygon points.

    origin : Point
        The new origin the polygons points shall be relative to.

    downsample : int, optional
        Layer downsample >= 1 (Default: 1)

    Returns
    -------
    tuple(Point)
        New polygon with points relative to origin.
    """
    rel_polygon = []
    for point in polygon:
        rel_polygon.append(Point((point.x - origin.x) / downsample,
                                 (point.y - origin.y) / downsample))

    return tuple(rel_polygon)