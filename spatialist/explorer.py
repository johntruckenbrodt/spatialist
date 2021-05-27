"""
Visualization tools using Jupyter notebooks
"""
import os
import re
import math
import inspect
import numpy as np
from .raster import Raster
from .vector import Vector
import matplotlib.pyplot as plt
from osgeo import ogr

try:
    from tkinter import filedialog, Tk
    from IPython.display import display
    from ipywidgets import interactive_output, IntSlider, Layout, Checkbox, Button, HBox, Label, VBox
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    raise RuntimeError('this module requires installation of the optional visualization requirements')


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
        the color map name for displaying the image.
        See :class:`matplotlib.colors.Colormap`.
    band_indices: list or None
        a list of indices for renaming the individual band indices in `filename`;
        e.g. -70:70, instead of the raw band indices, e.g. 1:140.
        The number of unique elements must be of same length as the number of bands in `filename`.
    band_names: list or None
        alternative names to assign to the individual bands
    pmin: int
        the minimum percentile for linear histogram stretching
    pmax: int
        the maximum percentile for linear histogram stretching
    zmin: int or float or None
        the minimum value of the displayed data range; overrides `pmin`
    zmax: int or float or None
        the maximum value of the displayed data range; overrides `pmax`
    ts_convert: function or None
        a function to read time stamps from the band names
    title: str or None
        the plot title to be displayed; per default, if set to `None`: `Figure 1`, `Figure 2`, ...
    datalabel: str
        a label for the units of the displayed data. This also supports LaTeX mathematical notation.
        See `Text rendering With LaTeX <https://matplotlib.org/users/usetex.html>`_.
    spectrumlabel: str
        a label for the x-axis of the vertical spectra
    fontsize: int
        the label text font size
    custom: list or None
        Custom functions for plotting figures in additional subplots.
        Each figure will be updated upon click on the major map display.
        Each function is required to take at least an argument `axis`.
        Furthermore, the following optional arguments are supported:
        
            * `values` (:py:obj:`list`): the time series values collected from the last click
            * `timestamps` (:py:obj:`list`): the time stamps as returned by `ts_convert`
            * `band` (:py:obj:`int`): the index of the currently displayed band
            * `x` (:py:obj:`float`): the x map coordinate in units of the image CRS
            * `y` (:py:obj:`float`): the y map coordinate in units of the image CRS
        
        Additional subplots are automatically added in a row-major order.
        The list may contain `None` elements to leave certain subplots empty for later usage.
        This might be useful for plots which are not to be updated each time the map display is clicked on.

    See Also
    --------
    :func:`matplotlib.pyplot.imshow`
    """
    
    def __init__(self, filename, cmap='jet', band_indices=None, band_names=None, pmin=2, pmax=98, zmin=None, zmax=None,
                 ts_convert=None, title=None, datalabel='data', spectrumlabel='time', fontsize=8, custom=None):
        
        self.ts_convert = ts_convert
        self.custom = custom
        
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
            if band_names is None:
                self.bandnames = ras.bandnames
            else:
                self.bandnames = band_names
            self.slider_readout = False
        
        self.timestamps = range(0, self.bands) if ts_convert is None else [ts_convert(x) for x in self.bandnames]
        
        self.datalabel = datalabel
        self.spectrumlabel = spectrumlabel
        
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
        self.zmin, self.zmax = zmin, zmax
        
        # define some options for display of the widget box
        self.layout = Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='100%'
        )
        self.fontsize = fontsize
        self.colormap = cmap
        
        if band_indices is not None:
            if len(list(set(band_indices))) != self.bands:
                raise RuntimeError('length mismatch of unique provided band indices ({0}) '
                                   'and image bands ({1})'.format(len(band_indices), self.bands))
            else:
                self.indices = band_indices
        else:
            self.indices = range(1, self.bands + 1)
        
        self.band = self.indices[len(self.indices) // 2]
        
        # define a slider for changing a plotted image
        self.slider = IntSlider(min=min(self.indices), max=max(self.indices), step=1, continuous_update=False,
                                value=self.band,
                                description='band',
                                style={'description_width': 'initial'},
                                readout=self.slider_readout)
        
        # a simple checkbox to enable/disable stacking of vertical profiles into one plot
        self.checkbox = Checkbox(value=True, description='stack vertical profiles', indent=False)
        
        # a button to clear the vertical profile plot
        self.clearbutton = Button(description='clear vertical plot')
        self.clearbutton.on_click(lambda x: self.__init_vertical_plot())
        
        self.write_csv = Button(description='export csv')
        self.write_csv.on_click(lambda x: self.csv())
        
        self.write_shp = Button(description='export shp')
        self.write_shp.on_click(lambda x: self.shp())
        
        if self.format == 'ENVI':
            self.sliderlabel = Label(value=self.bandnames[self.slider.value], layout={'width': '500px'})
            children = [HBox([self.slider, self.sliderlabel]),
                        HBox([self.checkbox, self.clearbutton, self.write_csv, self.write_shp])]
        else:
            children = [self.slider, HBox([self.checkbox, self.clearbutton, self.write_csv, self.write_shp])]
        
        form = VBox(children=children, layout=self.layout)
        
        display(form)
        
        # self.fig = plt.figure(num=title)
        if custom is None:
            self.fig, axes = plt.subplots(1, 2, num=title)
            
            # left display (image)
            self.ax1 = axes[0]
            # right display (time series)
            self.ax2 = axes[1]
        else:
            rows = math.ceil(len(custom) / 2) + 1
            self.fig, axes = plt.subplots(rows, 2, num=title)
            self.ax1, self.ax2 = axes[0]
            self.cax = np.ravel(axes[1:])
        
        self.ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        
        self.ax1.set_xlabel(self.xlab, fontsize=self.fontsize)
        self.ax1.set_ylabel(self.ylab, fontsize=self.fontsize)
        
        self.ax1.tick_params(axis='both', which='major', labelsize=self.fontsize)
        self.ax2.tick_params(axis='both', which='major', labelsize=self.fontsize)
        
        # format the values displayed for the mouse pointer
        self.ax1.format_coord = self.__format_coord
        
        # add a cross-hair to the horizontal slice plot
        self.x_coord, self.y_coord = self.__img2map(0, 0)
        self.lhor = self.ax1.axhline(self.y_coord, linewidth=1, color='r')
        self.lver = self.ax1.axvline(self.x_coord, linewidth=1, color='r')
        
        # set up the vertical profile plot
        self.__init_vertical_plot()
        
        # make the figure responds to mouse clicks by executing method __onclick
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        
        # enable interaction with the slider
        out = interactive_output(self.__onslide, {'h': self.slider})
        
        plt.tight_layout()
    
    def __onslide(self, h):
        self.band = self.indices.index(h)
        mat = self.__read_band(self.band + 1)
        masked = np.ma.array(mat, mask=np.isnan(mat))
        pmin, pmax = np.percentile(masked.compressed(), (self.pmin, self.pmax))
        vmin = self.zmin if self.zmin is not None else pmin
        vmax = self.zmax if self.zmax is not None else pmax
        cmap = plt.get_cmap(self.colormap)
        cmap.set_bad('white')
        title = self.bandnames[self.slider.value - 1]
        self.ax1.set_title(title, fontsize=self.fontsize)
        self.ax1.imshow(masked, vmin=vmin, vmax=vmax, extent=self.extent, cmap=cmap)
        if hasattr(self, 'sliderlabel'):
            self.sliderlabel.value = title
        self.__set_colorbar(self.ax1)
        self.vline.set_xdata(self.timestamps[self.slider.value])
    
    def __read_band(self, band):
        with Raster(self.filename) as ras:
            mat = ras.matrix(band)
        return mat
    
    def __img2map(self, x, y):
        x_map = self.xmin + self.xres * x
        y_map = self.ymax - self.yres * y
        return x_map, y_map
    
    def __map2img(self, x, y):
        x_img = int((x - self.xmin) / self.xres)
        y_img = int((self.ymax - y) / self.yres)
        return x_img, y_img
    
    def __reset_crosshair(self):
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
        self.lhor.set_ydata(self.y_coord)
        self.lver.set_xdata(self.x_coord)
    
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
        self.ax2.set_ylabel(self.datalabel, fontsize=self.fontsize)
        self.ax2.set_xlabel(self.spectrumlabel, fontsize=self.fontsize)
        self.ax2.set_title('vertical point profiles', fontsize=self.fontsize)
        self.ax2.set_xlim([min(self.timestamps), max(self.timestamps)])
        # plot vertical line at the slider position
        self.vline = self.ax2.axvline(self.timestamps[self.slider.value], color='black')
    
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
            self.__reset_crosshair()
            
            # read the time series at the clicked coordinate
            with Raster(self.filename)[self.y_coord, self.x_coord, :] as ras:
                timeseries = ras.array()
            
            # convert the map coordinates collected at the click to image pixel coordinates
            x, y = self.__map2img(self.x_coord, self.y_coord)
            
            # redraw/clear the vertical profile plot in case stacking is disabled
            if not self.checkbox.value:
                self.__init_vertical_plot()
            
            # plot the vertical profile
            label = 'x: {0:03}; y: {1:03}'.format(x, y)
            self.ax2.plot(self.timestamps, timeseries, label=label)
            self.ax2_legend = self.ax2.legend(loc=0, prop={'size': 7}, markerscale=1)
            if self.custom is not None:
                for i, func in enumerate(self.custom):
                    if func is not None:
                        self.cax[i].cla()
                        args = self.__argcheck(function=func,
                                               axis=self.cax[i],
                                               values=timeseries)
                        func(**args)
            plt.tight_layout()
    
    def csv(self, outname=None):
        """
        write the collected samples to a CSV file
        
        Parameters
        ----------
        outname: str
            the name of the file to write; if left at the default `None`, a graphical file selection dialog is opened

        Returns
        -------

        """
        # the first line is the vertical band line and is thus excluded
        profiles = self.ax2.get_lines()[1:]
        if len(profiles) == 0:
            return
        
        if outname is None:
            root = Tk()
            # Hide the main window
            root.withdraw()
            outname = filedialog.asksaveasfilename(initialdir=os.path.expanduser('~'),
                                                   defaultextension='.csv',
                                                   filetypes=(('csv', '*.csv'),
                                                              ('all files', '*.*')))
            if outname is None:
                return
        
        with open(outname, 'w') as csv:
            csv.write('id;bandname;row;column;xdata;ydata\n')
            for i in range(0, len(profiles)):
                line = profiles[i]
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                
                # get the row and column indices of the profile
                legend_text = self.ax2.get_legend().texts[i].get_text()
                legend_items = re.sub('[xy: ]', '', legend_text).split(';')
                col, row = [int(x) for x in legend_items]
                
                for j in range(0, self.bands):
                    entry = '{};{};{};{};{};{}\n'.format(i + 1, self.bandnames[j], row, col, xdata[j], ydata[j])
                    csv.write(entry)
            csv.close()
    
    def get_current_profile(self):
        """
        
        Returns
        -------
        list
            the values of the most recently plotted time series
        """
        profiles = self.ax2.get_lines()[1:]
        if len(profiles) == 0:
            return []
        else:
            line = profiles[-1]
            return line.get_ydata()
    
    def shp(self, outname=None):
        """
        write the collected samples to a CSV file

        Parameters
        ----------
        outname: str
            the name of the file to write; if left at the default `None`, a graphical file selection dialog is opened

        Returns
        -------

        """
        # the first line is the vertical band line and is thus excluded
        profiles = self.ax2.get_lines()[1:]
        if len(profiles) == 0:
            return
        
        if outname is None:
            root = Tk()
            # Hide the main window
            root.withdraw()
            outname = filedialog.asksaveasfilename(initialdir=os.path.expanduser('~'),
                                                   defaultextension='.shp',
                                                   filetypes=(('shp', '*.shp'),
                                                              ('all files', '*.*')))
            if outname is None:
                return
        
        layername = os.path.splitext(os.path.basename(outname))[0]
        
        with Vector(driver='Memory') as points:
            points.addlayer(layername, self.crs, 1)
            fieldnames = ['b{}'.format(i) for i in range(0, self.bands)]
            for field in fieldnames:
                points.addfield(field, ogr.OFTReal)
            
            for i, line in enumerate(profiles):
                # get the data values from the profile
                ydata = line.get_ydata().tolist()
                
                # get the row and column indices of the profile
                legend_text = self.ax2.get_legend().texts[i].get_text()
                legend_items = re.sub('[xy: ]', '', legend_text).split(';')
                col, row = [int(x) for x in legend_items]
                
                # convert the pixel indices to map coordinates
                x, y = self.__img2map(col, row)
                
                # create a new point geometry
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(x, y)
                fields = {}
                # create a field lookup dictionary
                for j, value in enumerate(ydata):
                    if np.isnan(value):
                        value = -9999
                    fields[fieldnames[j]] = value
                
                # add the new feature to the layer
                points.addfeature(point, fields=fields)
                point = None
            points.write(outname, 'ESRI Shapefile')
        lookup = os.path.splitext(outname)[0] + '_lookup.csv'
        with open(lookup, 'w') as csv:
            content = [';'.join(x) for x in zip(fieldnames, self.bandnames)]
            csv.write('id;bandname\n')
            csv.write('\n'.join(content))
    
    def __set_colorbar(self, axis, label=None):
        if len(axis.images) > 1:
            axis.images[0].colorbar.remove()
            del axis.images[0]
        
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        self.cbar = self.fig.colorbar(axis.images[0], cax=cax)
        self.cbar.ax.tick_params(axis='both', which='major', labelsize=self.fontsize)
        if label is not None:
            self.cbar.ax.set_ylabel(label, fontsize=self.fontsize)
    
    def __argcheck(self, function, axis, values):
        args = locals()
        del args['function']
        args['timestamps'] = self.timestamps
        args['band'] = self.band
        args['x'] = self.x_coord
        args['y'] = self.y_coord
        fargs = inspect.getfullargspec(function).args
        for required in ['axis']:
            if required not in fargs:
                raise TypeError("missing argument '{}'".format(required))
        return {key: value for key, value in args.items() if key in fargs}
    
    def __format_coord(self, x, y):
        text_pointer = 'x, y: {0}, {1}; ' \
                       + self.xlab + ', ' + self.ylab \
                       + ': {2:.2f}, {3:.2f}; value:'
        x_img, y_img = self.__map2img(x, y)
        return text_pointer.format(x_img, y_img, x, y)
