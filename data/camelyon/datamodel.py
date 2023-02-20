import os
import csv
from collections import defaultdict, OrderedDict, namedtuple
from typing import Tuple, Sequence, Any

from PIL import Image
import openslide
import xml.etree.ElementTree as Xml

from data.camelyon.cam_utils import Point, get_relative_polygon, draw_polygon, find_files

_RawAnnotation = namedtuple('RawAnnotation', 'name type_ part_of_group color polygon')

class Annotation:
    """Annotation class to provide access to a tumor annotation.

    Annotations can be displayed as an image with the annotation polygon put over the
    annotated section.


    Attributes
    ----------
    slide : Slide
        Slide the annotation belongs to.

    name : str
        Name of the annotation.

    type_ : str
        The type of the annotation specified in the annotation file.

    part_of_group: str
        The group of the annotation specified in the annotation file.

    color : tuple of int or str
        Annotation color as specified in the annotation file.

    polygon : sequence of Point
        A sequence of points annotating the tumor area.
    """

    def __init__(self, slide: 'Slide', name: str, type_: str, part_of_group: str,
                 color: Any, polygon: Sequence[Point]):
        """

        Parameters
        ----------
        slide : Slide
            Slide the annotation belongs to.

        name : str
            Name of the annotation.

        type_ : str
            The type of the annotation specified in the annotation file.

        part_of_group: str
            The group of the annotation specified in the annotation file.

        color : tuple of int or str
            Annotation color as specified in the annotation file.

        polygon : Sequence of Point
            A sequence of points annotating the tumor area.


        See Also
        --------
        PIL.ImageColor
        """
        self.slide = slide
        self.name = name
        self.type = type_
        self.part_of_group = part_of_group
        self.color = color
        self.polygon = polygon

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__,
            self.slide,
            self.name,
            self.type,
            self.part_of_group,
            self.color,
            self.polygon
        )

    def __str__(self):
        return '{}(slide={!r}, name={!r}, polygon size={!r})'.format(
            type(self).__name__,
            self.slide.name,
            self.name,
            len(self.polygon)
        )

    def get_boundaries(self, level, padding=0):
        """
        Return the annotation boundaries.

        Parameters
        ----------
        level : int
            Layer

        padding : int, optional
            Add additional pixels to the boundaries of the Annotation. (Default: 0)


        Returns
        -------
        origin : (int, int)
            Coordinates of the top left corner of the annotation on the specified layer.

        size : (int, int)
            Annotation width and height on the specified layer.

        """
        x = int(min([p.x for p in self.polygon]) - padding)
        y = int(min([p.y for p in self.polygon]) - padding)
        width = int(max([p.x for p in self.polygon]) - x + padding)
        height = int(max([p.y for p in self.polygon]) - y + padding)

        downsample = self.slide.level_downsamples[level]

        origin = Point(x, y)
        size = (int(width / downsample), int(height / downsample))

        return origin, size

    def get_image(self, *, level=4, padding=100, fill=(50, 50, 50, 80)) -> Image.Image:
        """
        Create an image of the annotated tissue section overlayed with the annotation polygon.

        The polygon's outline `color` will be set to the color attribute of the
        `Annotation` itself. The `fill` color can be specified via the parameter `fill`.

        Parameters
        ----------
        level : int, optional
            Slide level/layer used to create the image.

        padding : int, optional
            Padding added to either side of the image in pixel. Padding is added on layer
            0 and will be downsacled if a `level` higher than 0 is passed.

        fill : tuple of int or str, optional
            Annotation color used to fill the polygon.
            (Default: (50, 50, 50, 80), a dark gray).

        Returns
        -------
        Image.Image
            Image picturing the annotated section from the slide with annotation overlay.

        See Also
        --------
        PIL.ImageColor
        """
        origin, image_size = self.get_boundaries(level, padding)
        downsample = self.slide.level_downsamples[level]

        return draw_polygon(self.slide.read_region(origin, level, image_size),
                            get_relative_polygon(self.polygon, origin,
                                                 downsample),
                            fill=fill,
                            outline=self.color)

def _get_raw_annotations(filename):
    """
    Read all annotation data from an ASAP XML file.

    Parameters
    ----------
    filename : str
        File name of the annotation XML-File.

    Returns
    -------
    Tuple[_RawAnnotation]
        Parsed annotation form XML-File.
    """
    #logger.debug('Reading annotation data from {}', filename)
    tree = Xml.parse(filename)
    root = tree.getroot()
    annotations = []

    for annotation in root.iter('Annotation'):
        # all annotation points sorted by the `Order` attribute
        polygon = (Point(float(c.attrib['X']), float(c.attrib['Y'])) for c in
                   sorted(annotation.iter('Coordinate'),
                          key=lambda x: int(x.attrib['Order'])))

        annotations.append(_RawAnnotation(
            annotation.attrib['Name'].replace(' ', ''),
            annotation.attrib['Type'],
            annotation.attrib['PartOfGroup'],
            annotation.attrib['Color'],
            tuple(polygon)
        ))

    return tuple(annotations)

class Slide(openslide.OpenSlide):
    """
    Wrapper class for openslide.OpenSlide.

    In addition to the OpenSlide itself this class holds information like name and
    possible annotations and stage of the slide itself.

    Attributes
    ----------
    name : str
        Name of the slide.

    stage : str or None
        pN-stage of the slide (None for CAMELYON16 slides).

    has_tumor : bool
        True if the slide has annotations or a non negative pN-stage.

    is_annotated : bool
        True if the slide has annotation.

    See Also
    --------
    openslide.OpenSlide
    """

    def __init__(self, name, filename, annotation_filename=None, stage=None,
                 otsu_thresholds=None):
        """
        Parameters
        ----------
        name : str
            Slide name. Usually the filename without extension.

        filename : str
            Relative or absolute path to slide file.

        annotation_filename : str or None, optional
            Relative or absolute path to an annotation XML file. (Default: None)

        stage : str or None, optional
            nP-stage for CAMELYON17 slides. Leave `None` for CAMELYON16 slides.
            (Default: None)

        otsu_thresholds : dict of float or None, optional
            Dictionary with otsu thresholds for each level. (Default: None)
            Dictionary does not have to be exhaustive e.g.: {0: 6.333, 5: 7.0}
        """
        super().__init__(filename)
        self.name = name
        self._filename = filename
        self._annotation_filename = annotation_filename
        self.stage = stage
        self.is_annotated = self._annotation_filename is not None
        self.has_tumor = self.is_annotated or (
            self.stage is not None and self.stage != 'negative')
        self._otsu_thresholds = otsu_thresholds if otsu_thresholds is not None else {}
        self._annotations = None

    @property
    def annotations(self) -> Tuple[Annotation]:
        """
        Return a tuple of all annotations.

        Returns
        -------
        tuple of Annotation
            All annotations belonging to this instance of `Slide` as a tuple.
        """
        if self._annotations is None:
            if self.is_annotated:
                raw_annotations = _get_raw_annotations(self._annotation_filename)
                self._annotations = tuple(Annotation(self, *x) for x in raw_annotations)
            else:
                self._annotations = ()

        return self._annotations

    def get_full_slide(self, level) -> Image.Image:
        """
        Return the full image of a slide layer.

        Returns
        -------
        Image.Image
            Complete slide on layer `level`.
        """
        return self.read_region((0, 0), level, self.level_dimensions[level])

    def get_otsu_threshold(self, level):
        """
        Return pre-calculated otsu threshold of a layer.

        Parameters
        ----------
        level : int
            Slide layer

        Returns
        -------
        otsu_threshold: float or None
            Otsu threshold of layer `level` or None if not pre-calculated.
        """
        if level in self._otsu_thresholds:
            return self._otsu_thresholds[level]
        else:
            return None

    def __repr__(self):
        if self.is_annotated:
            repr_str = "{}({!r}, {!r}, {!r}, {!r})"
        else:
            repr_str = "{}({!r}, {!r}, {!r})"

        return repr_str.format(type(self).__name__,
                               self.name,
                               self._filename,
                               self.stage,
                               self._annotation_filename)

class SlideManager:
    """
    Provide access to slices from CAMELYON16.

    Attributes
    ----------
    negative_slides : tuple of Slide
        All slides that do not have annotations.

    annotated_slides : tuple of Slide
        All slides that have annotations.
    """

    def __init__(self, *, data_dir, otsu_fname):
        """
        Initialize the CAMELYON data set.

        Parameters
        ----------
        data_dir : str
            Path to the CAMELYON16 directory.
        """

        self._slides = OrderedDict()
        self.slide_paths = OrderedDict()
        self.annotation_paths = OrderedDict()
        self.stages = OrderedDict()
        self.negative_slides = tuple()
        self.annotated_slides = tuple()
        self.test_slides = tuple()

        self.num_positive_train = 0
        self.num_negative_train = 0

        data_dir = os.path.expanduser(data_dir)
        self._path = {
            'dir': data_dir,
            'negative': os.path.join(data_dir, 'training/normal'),
            'positive': os.path.join(data_dir, 'training/tumor'),
            'annotations': os.path.join(data_dir, 'training/lesion_annotations'),
            'test': os.path.join(data_dir, 'testing/images'),
            'test_annotations': os.path.join(data_dir, 'testing/lesion_annotations'),
            'otsu': os.path.join(data_dir, otsu_fname)
        }
        self.__load_data()

    def __load_data(self):
        """Load slides."""

        # Negative slides
        self.otsu_thresholds = defaultdict(dict)
        try:
            with open(self._path['otsu'], 'r') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    self.otsu_thresholds[line['name']][int(line['level'])] = float(
                        line['threshold'])
        except FileNotFoundError:
            print('No pre-calculated otsu thresholds found.')

        slide_files = find_files('*.tif', self._path['negative'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            slide = Slide(slide_name, slide_path,
                          otsu_thresholds=self.otsu_thresholds[slide_name])

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.negative_slides += (slide,)
            self.num_negative_train += 1

        # Positive (tumor) slides
        slide_files = find_files('*.tif', self._path['positive'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            annotation_path = os.path.join(self._path['annotations'],
                                           f'{slide_name}.xml')
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(annotation_path)
            slide = Slide(slide_name, slide_path, otsu_thresholds=self.otsu_thresholds[slide_name],
                annotation_filename=annotation_path)

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.annotation_paths[slide_name] = annotation_path
            self.annotated_slides += (slide,)
    
            self.num_positive_train += 1

        # test slides
        slide_files = find_files('*.tif', self._path['test'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            annotation_path = os.path.join(self._path['test_annotations'],
                                           f'{slide_name}.xml')
            if not os.path.exists(annotation_path):
                slide = Slide(slide_name, slide_path,
                          otsu_thresholds=self.otsu_thresholds[slide_name])
            else:
                slide = Slide(slide_name, slide_path, otsu_thresholds=self.otsu_thresholds[slide_name], 
                    annotation_filename=annotation_path)
                
                self.annotation_paths[slide_name] = annotation_path

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.test_slides += (slide,)


    @property
    def slides(self) -> Tuple[Slide]:
        """
        Return all slides as tuple.

        Returns
        -------
        tuple of Slide
            All slides managed by the instance of `SlideManager`.
        """
        return tuple(self._slides.values())

    @property
    def slide_names(self) -> Tuple[str]:
        """
        Return slide names as tuple.

        Returns
        -------
        tuple of str
            Slide names of all slides managed by the instance of `SlideManager`.
        """
        return tuple(self._slides.keys())

    def get_slide_names_subset(self, train=True) -> Tuple[str]:
        """
        Return slide names as tuple.

        Returns
        -------
        tuple of str
            Slide names of all slides managed by the instance of `SlideManager`.
        """
        if train:
            names = tuple(name for name in self._slides.keys() if 'test' not in name)
        else:
            names = tuple(name for name in self._slides.keys() if 'test' in name)

        return names

    def get_slide(self, name) -> Slide:
        """
        Retrieve a slide by its name.

        Parameters
        ----------
        name : str
            Slide name.


        Returns
        -------
        Slide
            Slide-Object with the name passed.
        """
        return self._slides[name]

    def __repr__(self):
        return '{}(cam16_dir={!r}, cam17_dir={!r})'.format(type(self).__name__,
                                                           self._path['cam16']['dir'],
                                                           self._path['cam17']['dir'])

    def __str__(self):
        return 'SlideManager contains: {} Slides ({} annotated; {} negative)'.format(
            len(self.slides),
            len(self.annotated_slides),
            len(self.negative_slides))