import numpy as np
from .raster import Raster
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interactive_output, IntSlider, Layout, Checkbox, Button, HBox


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
        a list of indices for renaming the individual band indices in `filename`;
         e.g. -70:70, instead of the raw band indices, e.g. 1:140.
        The number of unique elements must of same length as the number of bands in `filename`.
    """

    def __init__(self, filename, cmap='jet', band_indices=None, pmin=2, pmax=98):

        matplotlib.rcParams['figure.figsize'] = (13, 4)

        self.filename = filename
        with Raster(filename) as ras:
            self.rows = ras.rows
            self.cols = ras.cols
            self.bands = ras.bands
            self.epsg = ras.epsg
            self.crs = ras.srs
            geo = ras.raster.GetGeoTransform()

        xlab = self.crs.GetAxisName(None, 0)
        ylab = self.crs.GetAxisName(None, 1)
        self.xlab = xlab.lower() if xlab is not None else 'longitude'
        self.ylab = ylab.lower() if ylab is not None else 'latitude'

        self.xmin = geo[0]
        self.ymax = geo[3]
        self.xres = geo[1]
        self.yres = abs(geo[5])

        self.xmax = self.xmin + self.xres * self.cols
        self.ymin = self.ymax - self.yres * self.rows

        self.extent = (self.xmin, self.xmax, self.ymin, self.ymax)

        self.pmin, self.pmax = pmin, pmax

        # define some options for display of the widget box
        self.layout = Layout(
            display='flex',
            flex_flow='row',
            border='solid 2px',
            align_items='stretch',
            width='100%'
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
                                style={'description_width': 'initial'})

        # a simple checkbox to enable/disable stacking of vertical profiles into one plot
        self.checkbox = Checkbox(value=True, description='stack vertical profiles', indent=False)

        # a button to clear the vertical profile plot
        self.clearbutton = Button(description='clear vertical plot')
        self.clearbutton.on_click(lambda x: self.__init_vertical_plot())

        form = HBox(children=[self.slider, self.checkbox, self.clearbutton],
                    layout=self.layout)

        display(form)

        self.fig = plt.figure()

        # display of SLC amplitude
        self.ax1 = self.fig.add_subplot(121)
        # display of topographical phase
        self.ax2 = self.fig.add_subplot(122)

        # self.ax1 = plt.gca()
        self.ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax1.get_yaxis().get_major_formatter().set_useOffset(False)

        self.ax1.set_xlabel(self.xlab, fontsize=12)
        self.ax1.set_ylabel(self.ylab, fontsize=12)

        text_pointer = self.ylab + '={0:.2f}, ' + self.xlab + '={1:.2f}, value='
        self.ax1.format_coord = lambda x, y: text_pointer.format(y, x)

        # add a cross-hair to the horizontal slice plot
        self.x_coord, self.y_coord = self.img2map(0, 0)
        self.lhor = self.ax1.axhline(self.y_coord, linewidth=1, color='r')
        self.lver = self.ax1.axvline(self.x_coord, linewidth=1, color='r')

        # set up the vertical profile plot
        self.__init_vertical_plot()

        # make the figure respond to mouse clicks by executing method __onclick
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onclick)

        # enable interaction with the slider
        out = interactive_output(self.__onslide, {'h': self.slider})

        plt.tight_layout()

    def __onslide(self, h):
        mat = self.__read_band(self.indices.index(h) + 1)
        pmin, pmax = np.percentile(mat, (self.pmin, self.pmax))
        self.ax1.imshow(mat, vmin=pmin, vmax=pmax, extent=self.extent, cmap=self.colormap)

    def __read_band(self, band):
        with Raster(self.filename) as ras:
            mat = ras.matrix(band)
        return mat

    def read_timeseries(self, x, y):
        with Raster(self.filename) as ras:
            vals = ras.raster.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1)
        return vals.reshape(vals.shape[0])

    def img2map(self, x, y):
        x_map = self.xmin + self.xres * x
        y_map = self.ymax - self.yres * y
        return x_map, y_map

    def map2img(self, x, y):
        x_img = int((x - self.xmin) / self.xres)
        y_img = int((self.ymax - y) / self.yres)
        return x_img, y_img

    def __reset_crosshair(self, x, y):
        """
        redraw the cross-hair on the horizontal slice plot

        Parameters
        ----------
        x: int
            the x image coordinate
        y: int
            the y image coordinate

        Returns
        -------
        """
        self.lhor.set_ydata(y)
        self.lver.set_xdata(x)
        plt.draw()

    def __init_vertical_plot(self):
        """
        set up the vertical profile plot

        Returns
        -------
        """
        # clear the plot if lines have already been drawn on it
        if len(self.ax2.lines) > 0:
            self.ax2.cla()
        # set up the vertical profile plot
        self.ax2.set_ylabel('values', fontsize=12)
        self.ax2.set_xlabel('time', fontsize=12)
        self.ax2.set_title('vertical point profiles', fontsize=12)

    def __onclick(self, event):
        """
        respond to mouse clicks in the plot.
        This function responds to clicks on the first (horizontal slice) plot and updates the vertical profile and
        slice plots

        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            the click event object containing image coordinates

        """
        # only do something if the first plot has been clicked on
        if event.inaxes == self.ax1:

            # retrieve the click coordinates
            self.x_coord = event.xdata
            self.y_coord = event.ydata

            # redraw the cross-hair
            self.__reset_crosshair(self.x_coord, self.y_coord)

            x, y = self.map2img(self.x_coord, self.y_coord)
            subset_vertical = self.read_timeseries(x, y)

            # redraw/clear the vertical profile plot in case stacking is disabled
            if not self.checkbox.value:
                self.__init_vertical_plot()

            # plot the vertical profile
            label = 'x: {0:03}; y: {1:03}'.format(x, y)
            self.ax2.plot(range(0, self.bands), subset_vertical, label=label)
            self.ax2_legend = self.ax2.legend(loc=0, prop={'size': 7}, markerscale=1)
