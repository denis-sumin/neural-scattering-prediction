import math
from io import BytesIO

import matplotlib.pyplot as plt
import numpy
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import Flowable

from .difference_metrics import get_colormap


class PdfImage(Flowable):
    """PdfImage wraps the first page from a PDF file as a Flowable
    which can be included into a ReportLab Platypus document.
    Based on the vectorpdf extension in rst2pdf (http://code.google.com/p/rst2pdf/)"""

    @classmethod
    def from_matplotlib_figure(cls, figure):
        figure.tight_layout()
        imgdata = BytesIO()
        figure.savefig(imgdata, format="pdf")
        return cls(imgdata)

    def __init__(self, filename_or_object, width=None, height=None, kind="direct"):
        # If using StringIO buffer, set pointer to begining
        if hasattr(filename_or_object, "read"):
            filename_or_object.seek(0)
        page = PdfReader(filename_or_object, decompress=False).pages[0]
        self.xobj = pagexobj(page)
        self.imageWidth = width
        self.imageHeight = height
        x1, y1, x2, y2 = self.xobj.BBox

        self._w, self._h = x2 - x1, y2 - y1
        if not self.imageWidth:
            self.imageWidth = self._w
        if not self.imageHeight:
            self.imageHeight = self._h
        self.__ratio = float(self.imageWidth) / self.imageHeight
        if kind in ["direct", "absolute"] or width == None or height == None:
            self.drawWidth = width or self.imageWidth
            self.drawHeight = height or self.imageHeight
        elif kind in ["bound", "proportional"]:
            factor = min(float(width) / self._w, float(height) / self._h)
            self.drawWidth = self._w * factor
            self.drawHeight = self._h * factor

    def wrap(self, aW, aH):
        return self.drawWidth, self.drawHeight

    def drawOn(self, canv, x, y, _sW=0):
        if _sW > 0 and hasattr(self, "hAlign"):
            a = self.hAlign
            if a in ("CENTER", "CENTRE", TA_CENTER):
                x += 0.5 * _sW
            elif a in ("RIGHT", TA_RIGHT):
                x += _sW
            elif a not in ("LEFT", TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))

        xobj = self.xobj
        xobj_name = makerl(canv._doc, xobj)

        xscale = self.drawWidth / self._w
        yscale = self.drawHeight / self._h

        x -= xobj.BBox[0] * xscale
        y -= xobj.BBox[1] * yscale

        canv.saveState()
        canv.translate(x, y)
        canv.scale(xscale, yscale)
        canv.doForm(xobj_name)
        canv.restoreState()


def colored_histogram(
    image, difference_metric="rms", hist_width=128, hist_height=128, val_range=None
):
    if val_range is None:
        val_range = {"error": (-1.0, 1.0),}.get(
            difference_metric,
            (0.0, 20.0) if "ciede" in difference_metric else (0.0, 1.0),
        )

    colormap = get_colormap(difference_metric)

    hist_colors = numpy.tile(
        plt.get_cmap(colormap)(numpy.linspace(0, 1, hist_width)), (hist_height, 1, 1)
    )
    hist_x_idx, hist_y_idx = numpy.meshgrid(
        numpy.linspace(0, hist_width - 1, hist_width),
        numpy.linspace(hist_height - 1, 0, hist_height),
    )

    diff_hist = numpy.histogram(numpy.clip(image, *val_range), bins=hist_width, range=val_range)[
        0
    ]
    column_idx = ((diff_hist / numpy.max(diff_hist)) * hist_height).astype(numpy.int)
    diff_hist_img = numpy.zeros((hist_height, hist_width, 4))
    diff_hist_img[hist_y_idx < column_idx, :] = hist_colors[hist_y_idx < column_idx, :]
    return (
        diff_hist_img,
        val_range,
        (
            numpy.min(image),
            numpy.quantile(image, 0.1),
            numpy.mean(image),
            numpy.quantile(image, 0.9),
            numpy.max(image),
        ),
    )


def augment_diff_image(
    img,
    difference_metric,
    num_ticks=11,
    dpi=72,
    histogram_height=128,
    difference_values=None,
    difference_values_range=None,
    output_path=None,
):
    """
    Plot a colored histogram underneath the image
    """
    if difference_values is None:
        difference_values = img
    figsize = numpy.array(img.shape[:2]) / dpi  # .astype(int)
    hist_grid_height = int(math.ceil(histogram_height / dpi))
    figsize[1] += hist_grid_height * 1.25

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    # canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    gs1 = fig.add_gridspec(
        nrows=int(figsize[0]),
        ncols=int(figsize[1]),
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        hspace=0.00,
    )
    f_ax1 = fig.add_subplot(gs1[:-hist_grid_height, :])
    f_ax2 = fig.add_subplot(gs1[-hist_grid_height:, :])  # bottom row for the histogram

    f_ax1.imshow(img, aspect="equal")
    f_ax1.set_axis_off()

    hist, val_range, stats = colored_histogram(
        difference_values, difference_metric, val_range=difference_values_range
    )
    f_ax2.imshow(hist, aspect="auto")
    f_ax2.set_title(
        "{}\nMin:{:.2f}  10%:{:.2f}  Mean:{:.2f}  90%:{:.2f}  Max:{:.2f}".format(
            difference_metric, *stats
        ),
        y=0.8,
    )
    f_ax2.set_xticks(numpy.linspace(0, hist.shape[1], num_ticks))
    f_ax2.set_xticklabels(numpy.round(numpy.linspace(*val_range, num_ticks), decimals=2))
    f_ax2.set_yticks([])

    fig.canvas.draw()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    buffer, dims = fig.canvas.print_to_buffer()
    image_from_plot = numpy.frombuffer(buffer, dtype=numpy.uint8)
    image_from_plot = image_from_plot.reshape(*dims + (4,))
    return image_from_plot
