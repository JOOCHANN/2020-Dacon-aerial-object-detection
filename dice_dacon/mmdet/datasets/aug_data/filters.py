import numpy as np
import cv2
from .util import img_list_loader
from scipy.interpolate import UnivariateSpline

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"
def verify_alpha_channel(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame =cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

def apply_green(frame, alpha=0.37):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape

    d=1.9
    bright=1.5
    blue,green,red =80/d,250/d,60/d
    
    sepia_bgra= (blue, green, red, 1)
    overlay =np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, bright*alpha, frame, (1-alpha)*bright, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
def apply_night(frame, alpha=0.37):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape

    d=1.2
    bright=2
    blue,green,red =55/d,8/d,1/d
    
    sepia_bgra= (blue, green, red, 1)
    overlay =np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, bright*alpha, frame, (1-alpha)*bright, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def apply_fog(frame,fog, bright=1,alpha=0.45):
    if frame.shape[0] == fog.shape[0]:
        return cv2.addWeighted(fog, bright*alpha, frame, (1-alpha)*bright, 1)
    fog = cv2.resize(fog, frame.shape[:2])
    return cv2.addWeighted(fog, bright*alpha, frame, (1-alpha)*bright, 1)
    
# def apply_fog(frame,fog, bright=1,alpha=0.45):
    
#     return cv2.addWeighted(fog, bright*alpha, frame, (1-alpha)*bright, 1)
    
def resize_filters(img_dir,size=(3000,3000)):
    img_path_list= img_list_loader(img_dir,'png')
    for path in img_path_list:
        filter =cv2.imread(path)        
        filter=cv2.resize(filter,size)        
        cv2.imwrite(path,filter)      

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >0:
        lim_ceil = 255 - value
        v[v > lim_ceil] = 255
        v[v <= lim_ceil] += value
    else:
        lim_floor = 0 - value
        v[v < lim_floor] = 0
        v[v >= lim_floor] -= -value


    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

class CoolingFilter:
    """Warming filter
        A class that applies a warming filter to an image.
        The class uses curve filters to manipulate the perceived color
        temparature of an image. The warming filter will shift the image's
        color spectrum towards red, away from blue.
    """

    def __init__(self):
        """Initialize look-up table for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        """Applies warming filter to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)

    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))


class WarmingFilter:
    """Cooling filter
        A class that applies a cooling filter to an image.
        The class uses curve filters to manipulate the perceived color
        temparature of an image. The warming filter will shift the image's
        color spectrum towards blue, away from red.
    """

    def __init__(self):
        """Initialize look-up table for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)

    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))


class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """

    def __init__(self):
        pass

    def render(self, img_rgb):
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 7  # number of bilateral filtering steps

        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # make sure resulting image has the same dims as original
        img_color = cv2.resize(img_color, img_rgb.shape[:2])

        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(img_color, img_edge)