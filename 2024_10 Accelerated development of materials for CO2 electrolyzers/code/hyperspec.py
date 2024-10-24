from hyperspy_mod import BCF_reader
from typing import Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os


class HyperPos:
    """
    Processes hyperspectral data for a single map region (i.e.  position) from the Bruker
    Tornado M4.
    """

    def __init__(
            self,
            path: str,
            sid: Optional[int] = None,
            pid: Optional[int] = None,
            origin_x: Optional[float] = None,
            origin_y: Optional[float] = None,
    ):
        """
        Load in the BCF file from the Bruker Tornado M4.
        :param path: Path to BCF file.
        :param pid: The relative position ID of the spec in relation to the sample.
        :param origin_x: The origin of the slide (bottom left) in the eyes of the XRF.
        :param origin_y: The origin of the slide (bottom left) in the eyes of the XRF.
        """

        # Store the path
        self.path = path

        # Load the hyperspy object. This uses a slightly modified version of Hyperspy to
        # read in data from the Bruker M4.
        self.bcf = BCF_reader(path)

        # The sample and position of the Hyperspec
        self.pid: Optional[int] = pid
        self.sid: Optional[int] = sid

        # Integration boundaries used for quantifying specific elements. Values are in keV.
        self.bounds = dict(
            all=dict(int_min=-np.inf, int_max=np.inf),
            pd=dict(int_min=2.6, int_max=3.2),  # LA band
            cu=dict(int_min=7.9, int_max=8.2),  # KA band
            ag=dict(int_min=21.84, int_max=22.38),  # KA band peak is 22.16
            ag_la=dict(int_min=2.9, int_max=3.07),  # LA band
            co=dict(int_min=6.8, int_max=7.05),  # KA band
            au=dict(int_min=21.84, int_max=22.38),  # LA band
            al=dict(int_min=21.84, int_max=22.38),  # K band
            cl=dict(int_min=2.5, int_max=2.75),  # K band
            s=dict(int_min=2.2, int_max=2.4),  # K band
        )

        # Currently, the last hypermap row (lowest y value) is always 0. If the last row is
        # zeros, and self.trim_y is true, this row will be removed. The spacial size of the
        # map will also be adjusted. The left-post positional row is also troublesome.
        self.trim_x = True
        self.trim_y = True

        # Generate the data storage attributes

        # raw is the raw counts, cal is counts per second per eV.
        # array of shape y, x, c, where c is the length of the spectra.
        self.hmap_raw: Optional[np.ndarray] = None
        self.hmap_cal: Optional[np.ndarray] = None

        # The resolution (spatial, spatial, eV) of the map
        self.x_count: Optional[int] = None
        self.y_count: Optional[int] = None
        self.c_count: Optional[int] = None

        # The position of the map in a normal, right-handed, arbitrary (XRF) reference frame
        # TODO Something is wrong with these distances
        self.x0_mm: Optional[float] = None
        self.x1_mm: Optional[float] = None
        self.y0_mm: Optional[float] = None
        self.y1_mm: Optional[float] = None

        # Position of the map, in reference to the origin of the slide
        self.origin_x: Optional[float] = origin_x
        self.origin_y: Optional[float] = origin_y
        self.s_x0_mm: Optional[float] = None
        self.s_x1_mm: Optional[float] = None
        self.s_y0_mm: Optional[float] = None
        self.s_y1_mm: Optional[float] = None

        # The time spent on each pixel
        # TODO This time is likely in s not ms
        self.point_time_ms: Optional[float] = None

        # The energy bounds, range, and step size (usually 0.05 keV) for the energy
        self.en_min_kev: Optional[float] = None
        self.en_max_kev: Optional[float] = None
        self.en_range: Optional[float] = None
        self.en_step_kev: Optional[float] = None

        # Process the hyperspy file into readable data. This processes all hyperspectral and
        # metadata.
        self._process_bcf_file()

        # Generate calibrated data (cps per eV). This will generate self.hmap_cal, which
        # is agnostic to collection time.
        self._calc_calib_signal()


    def _process_bcf_file(self):
        """
        Test that the BCF file was processed correctly.
        :return: None
        """

        # Load the raw hypermap data (creates an array of shape y, x, c, where c is the
        # length of the spectra.
        self.hmap_raw = self.bcf.parse_hypermap(0, cutoff_at_kV=40)

        # Currently the hypermap from the API reports zeros for the last spatial line of the
        # hypermap. Detect if this is the case. If so, remove this row.
        if self.hmap_raw[-1, :].sum() == 0 and self.trim_y:

            # Cut off the last row
            self.hmap_raw = self.hmap_raw[:-1]

        # If trim x, trim x
        if self.trim_x:
            self.hmap_raw = self.hmap_raw[:, 1:]

        # Calculate the spatial and spectral dimensions for easy access.
        self.x_count = self.hmap_raw.shape[1]
        self.y_count = self.hmap_raw.shape[0]
        self.c_count = self.hmap_raw.shape[2]

        # Get meta data from the BCF header
        header_meta = self.bcf.header.gen_hspy_item_dict_basic()
        metadata = header_meta['original_metadata']
        stage_meta = metadata['Stage']
        micro_meta = metadata['Microscope']

        # Calculate the stage bounds
        self.x0_mm = stage_meta['X']
        self.x1_mm = self.x0_mm - round(micro_meta['DX'] * self.x_count, 10)
        self.y1_mm = stage_meta['Y']
        self.y0_mm = self.y1_mm - round(micro_meta['DY'] * self.y_count, 10)
        self.z_mm = stage_meta['Z']

        # If the origin of the slide has been passed, calculate the bounds relative to
        # the slide.
        if self.origin_x is not None and self.origin_y is not None:

            # Note that the calibration for x is reversed, as the axis for x is reversed
            # in the XRF. (Because that makes sense.) Also note that the slide origin
            # (which is defined by convention to be the bottom-left) is now in the top-right
            # of the XRF, as the UR-5e has rotated the slide by 180deg. To perform this
            # transformation, the length of the slide is assumed to be 76.2 mm.

            slide_length = 76.2
            self.s_x0_mm = round(self.x1_mm - self.origin_x, 2)
            self.s_x1_mm = round(self.x0_mm - self.origin_x, 2)
            self.s_y0_mm = round(slide_length - (self.y1_mm - self.origin_y), 2)
            self.s_y1_mm = round(slide_length - (self.y0_mm - self.origin_y), 2)

            # Rotate the data into the slide frame
            self.hmap_raw = np.flip(np.flip(self.hmap_raw, 1), 0)

        # Get dwell time per point
        self.point_time_ms = metadata['DSP Configuration']['PixelAverage'] / 1e6

        # Get energy min and max. Note that the min energy may be slightly negative.
        # This is likely some calibration artifact from the instrument.
        scale = self.bcf.header.spectra_data[0].scale
        offset = self.bcf.header.spectra_data[0].offset
        self.en_min_kev = offset
        self.en_max_kev = self.c_count * scale + offset

        # Calculate the energy for each spectral point
        self.en_range = np.linspace(self.en_min_kev, self.en_max_kev, self.c_count)

        # Calculate energy step size
        self.en_step_kev = (self.en_max_kev - self.en_min_kev) / self.c_count

    def _calc_calib_signal(self):
        """
        Calculate the calibrated XRF signal (in units of counts per second per eV) such that
        it can be integrated and agnostic to data collection time.
        :return: None
        """

        # TODO This is not equal to the counts per second per eV from the Bruker software.
        self.hmap_cal = self.hmap_raw / (self.en_step_kev * 1E3) / \
            (self.point_time_ms * 1E-3)

    def get_range_int(self, int_min: float, int_max: float):
        """
        Integrate the spectral data between two points.
        :param int_min: lower int bound
        :param int_max: upper int bound
        :return: Spacial data
        """

        # Make integration mask
        abool = (self.en_range > int_min) & (self.en_range < int_max)

        # Sum and return
        return np.sum(self.hmap_cal[:, :, abool], axis=2)

    def get_element_int(self, element: str):
        """
        Calculate the integration for a particular element.
        :param element: The two letter ID for the element. First letter capitalized.
        :return: Spatial data.
        """

        edict = self.bounds[element]
        return self.get_range_int(
            int_min=edict['int_min'],
            int_max=edict['int_max']
        )

    def make_element_map(
            self,
            element: str,
            path: Optional[str] = None,
            from_zero: bool = True,
    ):
        """
        Create and save a map of a particular element.
        :param element: element to map. Two letters, lower case.
        :param path: Path to save the image. If none, uses BCF directory.
        :param from_zero: Should the min color value be set to zero, or the map min.
        :return: None
        """

        # If no path passed, get BCF directory
        if path is None:
            path = f'{self.path.split(".")[0]}_{element}-map.png'

        # Get element data
        edata = self.get_element_int(element=element)

        # Get size ratio of frame
        figsize = (7, 6.5)

        # Create a figure
        figure: plt.Figure = plt.figure(figsize=figsize, dpi=500)
        axes = plt.Axes = figure.add_subplot()
        divider = make_axes_locatable(axes)
        caxes = divider.append_axes('right', size='5%', pad=0.05)

        # Calculate the minimum to use for the colorscale
        cmin = 0 if from_zero else edata.min()

        # Calc extent
        if self.origin_x is not None and self.origin_y is not None:
            extent = [
                self.s_x0_mm,
                self.s_x1_mm,
                self.s_y0_mm,
                self.s_y1_mm,
            ]

            # Format
            axes.set_xticks([self.s_x0_mm, self.s_x1_mm])
            axes.set_yticks([self.s_y0_mm, self.s_y1_mm])
            axes.set_xlabel('position\n(x, mm)', labelpad=-7)
            axes.set_ylabel('position\n(y, mm)', labelpad=-5)

        else:
            extent = None

        # Plot
        im = axes.imshow(
            edata,
            aspect='equal',
            vmin=cmin,
            cmap='inferno',
            extent=extent,
        )

        # Create colorbar
        bar = figure.colorbar(im, cax=caxes, orientation='vertical')
        bar.set_ticks([cmin, edata.max()])
        bar.ax.text(
            2, 0.5, 'cps per eV',
            rotation='vertical',
            transform=caxes.transAxes,
            horizontalalignment='left',
            verticalalignment='center',

        )

        # Add title
        axes.set_title(path.split("\\")[-1])

        # Format
        figure.subplots_adjust(
            left=0.15,
            right=0.87,
            top=0.98,
            bottom=0.05,
        )

        # correct for flipped image
        axes.invert_yaxis()
        axes.invert_xaxis()

        # Save
        figure.savefig(path)
        figure.clf()


    def get_physical_area(
            self,
            center_x: float,
            center_y: float,
            x_span: float,
            y_span: float,
            element: str,
    ):
        """
        Get the calib XRF data for a physical position, defined in slide coordinates.
        :param center_x: Center of ROI, mm
        :param center_y: Center of ROI, mm
        :param x_span: width, mm
        :param y_span: height, mm
        :param element: element string
        :return: Array of calib XRF in space.
        """

        # Get the bounds of the roi
        r_x0 = round(center_x - x_span / 2, 2)
        r_x1 = round(center_x + x_span / 2, 2)
        r_y0 = round(center_y - y_span / 2, 2)
        r_y1 = round(center_y + y_span / 2, 2)

        # Get the fractional positions of the bounds
        f_x0 = (r_x0 - self.s_x0_mm) / (self.s_x1_mm - self.s_x0_mm)
        f_x1 = (r_x1 - self.s_x0_mm) / (self.s_x1_mm - self.s_x0_mm)
        f_y0 = (r_y0 - self.s_y0_mm) / (self.s_y1_mm - self.s_y0_mm)
        f_y1 = (r_y1 - self.s_y0_mm) / (self.s_y1_mm - self.s_y0_mm)

        # Confirm that these bounds are within limits
        for f in [f_x0, f_x1, f_y0, f_y1]:
            if f < 0 or f > 1:
                raise ValueError('Requested ROI is outside the map.')

        # Get element map
        map = self.get_element_int(element=element)
        smap = map.shape

        # Generate the hymap positions
        m_x0 = round(f_x0 * smap[1])
        m_x1 = round(f_x1 * smap[1])
        m_y0 = round((1 - f_y1) * smap[0])
        m_y1 = round((1 - f_y0) * smap[0])

        # Return the selection
        return map[m_y0:m_y1, m_x0:m_x1]

    def generate_sample_frame(
            self,
            elements: Optional[List[str]] = None,
    ):
        """
        Generate a CSV describing the distribution of elements in the sample
        :param elements: List of elements to describe.
        :return: None
        """

        # If elements is not passed, generate for all available elements
        if elements is None:
            elements = list(self.hpos[0].bounds.keys())

        # Place to store parameters
        params = []

        # Get metadata
        p = dict(
            sample=self.sid,
            position=self.pid,
        )

        # For each element:
        for e in elements:

            # Get array
            emap = self.get_element_int(element=e)

            # Store metadata about array
            p[f'{e}_min'] = emap.min()
            p[f'{e}_max'] = emap.max()
            p[f'{e}_avg'] = emap.mean()
            p[f'{e}_std'] = emap.std()

            # Store
            params.append(p)

        # Convert to frame and store
        return pd.DataFrame(params)


class HyperSample:
    """
    Contains several HyperPos instances for a single sample.
    """

    def __init__(
            self,
            path: str,
            origin_x: Optional[float] = None,
            origin_y: Optional[float] = None,

    ):
        """
        Load in all BCF from a directory for a single sample.
        :param path: Directory path containing many BCF files.
        :param origin_x: The origin of the slide (bottom left) in the eyes of the XRF.
        :param origin_y: The origin of the slide (bottom left) in the eyes of the XRF.
        """

        # Store path
        self.path = path

        # Store origin
        self.origin_x: Optional[float] = origin_x
        self.origin_y: Optional[float] = origin_y

        # Create place to store Hyperspecs
        self.hpos: List[HyperPos] = list()

        # If XRF folder present
        if 'XRF' in os.listdir(path):

            # Store sample id
            self.sid = int(path.split('_')[-1])

            # Get directory
            xrf_path = os.path.join(path, 'XRF')

            # For each file
            for fname in sorted(os.listdir(xrf_path)):

                # If BCF file
                if fname.endswith('.bcf'):

                    # Get the pid and path
                    pid = int(fname.split('-')[-1].split('.')[0])
                    fpath = f'{xrf_path}/{fname}'

                    # Store instance
                    hspec = HyperPos(
                        path=fpath,
                        pid=pid,
                        origin_x=origin_x,
                        origin_y=origin_y,
                    )

                    # Store
                    self.hpos.append(hspec)

    def generate_sample_frame(
            self,
            elements: Optional[List[str]] = None,
    ):
        """
        Generate a CSV describing the distribution of elements in the sample
        :param elements: List of elements to describe.
        :return: None
        """

        # If elements is not passed, generate for all available elements
        if elements is None:
            elements = list(self.hpos[0].bounds.keys())

        # Place to store parameters
        params = []

        # For each hyspec
        for h in self.hpos:

            # Get metadata
            p = dict(
                sample=self.sid,
                position=h.pid,
            )

            # For each element:
            for e in elements:

                # Get array
                emap = h.get_element_int(element=e)

                # Store metadata about array
                p[f'{e}_min'] = emap.min()
                p[f'{e}_max'] = emap.max()
                p[f'{e}_avg'] = emap.mean()
                p[f'{e}_std'] = emap.std()

                # Store
                params.append(p)

        # Convert to frame and store
        return pd.DataFrame(params)

    def make_element_maps(
            self,
            element: str,
            path: Optional[str] = None,
            from_zero: bool = True,
    ):
        """
        Create and save maps of a particular element for each position.
        :param element: element to map. Two letters, lower case.
        :param path: Path to save the image. If none, uses BCF directory.
        :param from_zero: Should the min color value be set to zero, or the map min.
        :return: None
        """

        # Iterate through all positions.
        for hpos in self.hpos:
            hpos.make_element_map(
                element=element,
                path=path,
                from_zero=from_zero,
            )

    def make_sample_map(
            self,
            element: str,
            path: Optional[str] = None,
            from_zero: bool = True,
    ):
        """
        Make a plot of a single slide with all the positions visualized on it.
        :param element: Element to visualize.
        :param path: Path to save the image. If none, uses BCF directory.
        :param from_zero: Should the min color value be set to zero, or the map min.
        :return: None
        """

        # Plotting constants
        grey = '#555555'
        lw=1

        # Get path to save
        if path is None:
            path = f'{self.path}/XRF/{element}_map.png'

        # Slide dimensions
        slide_x = 25.7
        slide_y = 75.8

        # Make figure object
        figure: plt.Figure = plt.figure(
            figsize=(8, 7),
            dpi=500,
        )
        p_axis: plt.Axes = figure.add_subplot(1, 2, 1)
        m_axis: plt.Axes = figure.add_subplot(1, 2, 2)
        caxes = figure.add_axes([0.85, 0.25, 0.03, .5])  # x, y, w, h

        # Create place to store ranges and imshows
        p_ranges = []
        p_images = []

        # For each HyperPos
        for hpos in self.hpos:

            # Get data
            d = hpos.get_element_int(element=element)

            # Plot data
            image = m_axis.imshow(
                X=d,
                extent=(
                    hpos.s_x0_mm,
                    hpos.s_x1_mm,
                    hpos.s_y0_mm,
                    hpos.s_y1_mm,
                ),
                cmap='inferno',
            )

            # Store data
            p_ranges.append(d)
            p_images.append(image)

            # Plot number and bounds on p axis
            p_axis.add_patch(Rectangle(
                xy=(
                    hpos.s_x0_mm,
                    hpos.s_y0_mm,
                ),
                width=hpos.s_x1_mm - hpos.s_x0_mm,
                height=hpos.s_y1_mm - hpos.s_y0_mm,
                edgecolor=grey,
                facecolor='none',
                lw=lw,
                zorder=1
            ))

            # Add number
            p_axis.text(
                (hpos.s_x0_mm + hpos.s_x1_mm) / 2,
                (hpos.s_y0_mm + hpos.s_y1_mm) / 2,
                s=str(hpos.pid),
                horizontalalignment='center',
                verticalalignment='center',
                fontdict=dict(
                    color=grey,
                    fontsize=8,
                ),
                zorder=2,
            )

        # Get range mins
        rmin = min([p.min() for p in p_ranges])
        rmax = max([p.max() for p in p_ranges])

        # Set color ranges
        for img in p_images:
            img.set_clim(rmin, rmax)

        # Colorbar
        bar = figure.colorbar(mappable=p_images[0], cax=caxes, orientation='vertical')
        bar.set_ticks([rmin, rmax])
        bar.ax.text(
            1.5, 0.5, 'cps per eV',
            rotation='vertical',
            transform=caxes.transAxes,
            horizontalalignment='left',
            verticalalignment='center',
        )

        # For both axes
        for ax in [m_axis, p_axis]:

            # Plot rectangle for slide
            ax.add_patch(Rectangle(
                xy=(0, 0),
                width=slide_x,
                height=slide_y,
                edgecolor=grey,
                facecolor='none',
                lw=lw,
            ))

            # Format
            buffer = 5
            ax.set_xlim(-buffer, slide_x + buffer)
            ax.set_ylim(-buffer, slide_y + buffer)
            ax.set_xticks([0, slide_x])
            ax.set_yticks([0, slide_y])
            ax.set_xlabel('position\n (x, mm)', labelpad=-5)
            ax.set_ylabel('position\n (y, mm)', labelpad=-10)
            ax.set_aspect('equal')

            # Plot dotted centerlines
            largs = dict(
                lw=lw,
                linestyle=':',
                c=grey,
                zorder=0,
            )
            ax.plot(
                [slide_x/2]*2,
                [0, slide_y],
                **largs
            )
            ax.plot(
                [0, slide_x],
                [slide_y/2]*2,
                **largs
            )

        figure.subplots_adjust(
            left=0.07,
            right=0.82,
            top=0.98,
            hspace=0.,
        )

        # Save
        figure.savefig(path)
        figure.clf()
