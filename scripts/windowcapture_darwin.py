from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGNullWindowID
import Quartz as QZ
import Quartz.CoreGraphics as CG
import numpy as np
from PIL import Image
import typing as t
from typing import Optional
import os
from pathlib import Path

Matcher = t.Callable[[dict[str, t.Any]], bool]

def find_window(matcher: Matcher) -> list[dict]:
        window_list = QZ.CGWindowListCopyWindowInfo(
            QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
        )

        result = []

        for window in window_list:
            if matcher(window):
                result.append(window)

        return result[0]

def new_name_matcher(name: str) -> Matcher:
    def matcher(window: dict[str, t.Any]) -> bool:
        return window.get("kCGWindowName") == name

    return matcher

class WindowCapture:
    width = 0
    height = 0
    hwnd = None
    cropped_x = None
    cropped_y = None
    mon = None

    def __init__(self, window_title):
        self.hwnd = find_window(new_name_matcher(window_title))

    def get_screenshot(self, x_0_crop=None, y_0_crop=None, x_1_crop=None, y_1_crop=None):
        file_path = os.path.join(Path.cwd().parent, "images/window_capture.bmp")
        region = None
        window_id=self.hwnd["kCGWindowNumber"]
        if x_0_crop != None:
            region = list[x_0_crop, y_0_crop, x_1_crop, y_1_crop]
            window_id = None
        if window_id is not None and region is not None:
            raise ValueError("Only one of region or window_id must be specified")

        image: Optional[CG.CGImage] = None

        if region is not None:
            cg_region = None

            if region is None:
                cg_region = CG.CGRectInfinite
            else:
                cg_region = CG.CGRectMake(*region)

            image = CG.CGWindowListCreateImage(
                cg_region,
                CG.kCGWindowListOptionOnScreenOnly,
                CG.kCGNullWindowID,
                CG.kCGWindowImageDefault,
            )
        elif window_id is not None:
            image = CG.CGWindowListCreateImage(
                CG.CGRectNull,
                CG.kCGWindowListOptionIncludingWindow,
                window_id,
                CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
            )
        else:
            raise ValueError("Either region or window_id must be specified")

        if image is None:
            raise ValueError("Could not capture image")

        bpr = CG.CGImageGetBytesPerRow(image)
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)

        cg_dataprovider = CG.CGImageGetDataProvider(image)
        cg_data = CG.CGDataProviderCopyData(cg_dataprovider)

        np_raw_data = np.frombuffer(cg_data, dtype=np.uint8)

        final_image = Image.fromarray(np.lib.stride_tricks.as_strided(
            np_raw_data,
            shape=(height, width, 3),
            strides=(bpr, 4, 1),
            writeable=True,
        ))
        final_image.save(file_path, "BMP")

        return final_image       

    def get_window_resolution(self):
        bounds = self.hwnd["kCGWindowBounds"]
        width, height = int(bounds["Width"]), int(bounds["Height"])
        x, y = int(bounds["X"]), int(bounds["Y"])
        return f"{width}x{height}"
    
    def get_window(self):
        return self.hwnd
