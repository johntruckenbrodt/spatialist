from .raster import Raster
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interactive_output, IntSlider, Layout


class RasterViewer(object):
    """
    plotting utility for displaying a geocoded image stack file.

    On moving the slider, the band at the slider position is read from the file and displayed.

    Parameters
    ----------
    filename: str
        the name of the file to display
    cmap: str
        the color map for displaying the image.
        See `matplotlib.pyplot.imshow <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_
    band_indices: list
        a list of indices for renaming the individual bands in `filename` such that one can scroll trough the
        range of inversion heights, e.g. -70:70, instead of the raw band indices, e.g. 1:140.
        The number of unique elements must of same length as the number of bands in `filename`.
    """

    def __init__(self, filename, cmap='jet', band_indices=None):
        self.filename = filename
        with Raster(filename) as ras:
            self.rows = ras.rows
            self.cols = ras.cols
            self.bands = ras.bands
            self.epsg = ras.epsg
            geo = ras.raster.GetGeoTransform()

        xmin = geo[0]
        ymax = geo[3]
        xres = geo[1]
        yres = abs(geo[5])

        xmax = xmin + xres * self.cols
        ymin = ymax - yres * self.rows

        self.extent = (xmin, xmax, ymin, ymax)

        # define some options for display of the widget box
        self.layout = Layout(
            display='flex',
            flex_flow='row',
            border='solid 2px',
            align_items='stretch',
            width='88%'
        )

        self.colormap = cmap

        if band_indices is not None:
            if len(list(set(band_indices))) != self.bands:
                raise RuntimeError('length mismatch of unique provided band indices ({0}) '
                                   'and image bands ({1})'.format(len(band_indices), self.bands))
            else:
                self.indices = sorted(band_indices)
        else:
            self.indices = range(1, self.bands + 1)

        # define a slider for changing a plotted image
        self.slider = IntSlider(min=min(self.indices), max=max(self.indices), step=1, continuous_update=False,
                                value=self.indices[len(self.indices)//2],
                                description='band index',
                                style={'description_width': 'initial'},
                                layout=self.layout)

        display(self.slider)

        self.fig = plt.figure()
        self.ax = plt.gca()
        self.ax.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax.get_yaxis().get_major_formatter().set_useOffset(False)

        self.ax.format_coord = lambda x, y: 'easting={0:.2f}, northing={1:.2f}, reflectivity='.format(x, y)

        # enable interaction with the slider
        out = interactive_output(self.__onslide, {'h': self.slider})

    def __onslide(self, h):
        mat = self.__read_band(self.indices.index(h) + 1)
        self.ax.imshow(mat, extent=self.extent, cmap=self.colormap)

    def __read_band(self, band):
        with Raster(self.filename) as ras:
            mat = ras.matrix(band)
        return mat
