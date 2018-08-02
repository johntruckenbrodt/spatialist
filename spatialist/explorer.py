import os
import re
import numpy as np
from .raster import Raster
from .envi import HDRobject
import matplotlib
import matplotlib.pyplot as plt

import sys
if sys.version_info >= (3, 0):
    from tkinter import filedialog, Tk
else:
    from Tkinter import Tk
    import tkFileDialog as filedialog

from IPython.display import display
from ipywidgets import interactive_output, IntSlider, Layout, Checkbox, Button, HBox, Label, VBox
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
This module is intended for gathering functionalities for plotting spatial data with jupyter notebooks
"""


class RasterViewer(object):
    """
    | Plotting utility for displaying a geocoded image stack file.
    | On moving the slider, the band at the slider position is read from the file and displayed.
    | By clicking on the band image display, you can display time series profiles.
    | The collected profiles can be saved to a csv file.

    Parameters
    ----------
    filename: str
        the name of the file to display
    cmap: str
        the color map for displaying the image.
        See `matplotlib.pyplot.imshow <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_.
    band_indices: list or None
        a list of indices for renaming the individual band indices in `filename`;
        e.g. -70:70, instead of the raw band indices, e.g. 1:140.
        The number of unique elements must of same length as the number of bands in `filename`.
    pmin: int
        the minimum percentile for linear histogram stretching
    pmax: int
        the maximum percentile for linear histogram stretching
    ts_convert: function or None
        a function to read time stamps from the band names
    title: str or None
        the plot title to be displayed; per default, if set to `None`: `Figure 1`, `Figure 2`, ...
    datalabel: str
        a label for the units of the displayed data. This also supports LaTeX mathematical notation.
        See `Text rendering With LaTeX <https://matplotlib.org/users/usetex.html>`_.

    """

    def __init__(self, filename, cmap='jet', band_indices=None, pmin=2, pmax=98, ts_convert=None, title=None, datalabel='data'):

        self.ts_convert = ts_convert

        self.filename = filename
        with Raster(filename) as ras:
            self.rows = ras.rows
            self.cols = ras.cols
            self.bands = ras.bands
            self.epsg = ras.epsg
            self.crs = ras.srs
            geo = ras.raster.GetGeoTransform()
            self.nodata = ras.nodata
            self.format = ras.format
            if self.format == 'ENVI':
                self.bandnames = HDRobject(filename+'.hdr').band_names
                self.slider_readout = False

        self.timestamps = range(0, self.bands) if ts_convert is None else [ts_convert(x) for x in self.bandnames]

        self.datalabel = datalabel

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
            flex_flow='column',
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
                                description='band',
                                style={'description_width': 'initial'},
                                readout=self.slider_readout)

        # a simple checkbox to enable/disable stacking of vertical profiles into one plot
        self.checkbox = Checkbox(value=True, description='stack vertical profiles', indent=False)

        # a button to clear the vertical profile plot
        self.clearbutton = Button(description='clear vertical plot')
        self.clearbutton.on_click(lambda x: self.__init_vertical_plot())

        self.write_csv = Button(description='export csv')
        self.write_csv.on_click(lambda x: self.__csv())

        if self.format == 'ENVI':
            self.sliderlabel = Label(value=self.bandnames[self.slider.value], layout={'width': '500px'})
            children = [HBox([self.slider, self.sliderlabel]), HBox([self.checkbox, self.clearbutton, self.write_csv])]
        else:
            children = [self.slider, HBox([self.checkbox, self.clearbutton, self.write_csv])]

        form = VBox(children=children, layout=self.layout)

        display(form)

        self.fig = plt.figure(num=title)

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
        self.x_coord, self.y_coord = self.__img2map(0, 0)
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
        masked = np.ma.array(mat, mask=np.isnan(mat))
        pmin, pmax = np.percentile(masked.compressed(), (self.pmin, self.pmax))
        cmap = plt.get_cmap(self.colormap)
        cmap.set_bad('white')
        self.ax1.imshow(masked, vmin=pmin, vmax=pmax, extent=self.extent, cmap=cmap)
        self.sliderlabel.value = self.bandnames[self.slider.value]
        self._set_colorbar(self.ax1, self.datalabel)

    def __read_band(self, band):
        with Raster(self.filename) as ras:
            mat = ras.matrix(band)
        return mat

    def __read_timeseries(self, x, y):
        with Raster(self.filename) as ras:
            vals = ras.raster.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1)
            vals[vals == self.nodata] = np.nan
        return vals.reshape(vals.shape[0])

    def __img2map(self, x, y):
        x_map = self.xmin + self.xres * x
        y_map = self.ymax - self.yres * y
        return x_map, y_map

    def __map2img(self, x, y):
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
        self.ax2.set_ylabel(self.datalabel, fontsize=12)
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

            x, y = self.__map2img(self.x_coord, self.y_coord)
            subset_vertical = self.__read_timeseries(x, y)

            # redraw/clear the vertical profile plot in case stacking is disabled
            if not self.checkbox.value:
                self.__init_vertical_plot()

            # plot the vertical profile
            label = 'x: {0:03}; y: {1:03}'.format(x, y)
            self.ax2.plot(self.timestamps, subset_vertical, label=label)
            self.ax2_legend = self.ax2.legend(loc=0, prop={'size': 7}, markerscale=1)

    def __csv(self):
        profiles = self.ax2.get_lines()
        if len(profiles) == 0:
            return
        root = Tk()
        # Hide the main window
        root.withdraw()
        f = filedialog.asksaveasfile(initialdir=os.path.expanduser('~'), mode='w', defaultextension='.csv',
                                     filetypes=(('csv', '*.csv'), ('all files', '*.*')))
        if f is None:
            return
        f.write('id;bandname;row;column;xdata;ydata\n')
        for i in range(0, len(profiles)):
            line = profiles[i]
            xdata = line.get_xdata()
            ydata = line.get_ydata()

            col, row = [int(x) for x in re.sub('[xy: ]', '', self.ax2.get_legend().texts[i].get_text()).split(';')]

            for j in range(0, self.bands):
                entry = '{};{};{};{};{};{}\n'.format(i+1, self.bandnames[j], row, col, xdata[j], ydata[j])
                f.write(entry)
        f.close()

    def _set_colorbar(self, axis, label):
        if len(axis.images) > 1:
            axis.images[0].colorbar.remove()
            del axis.images[0]

        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cbar = self.fig.colorbar(axis.images[0], cax=cax)
        cbar.ax.set_ylabel(label, fontsize=12)
