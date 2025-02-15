import ctypes
from PIL import Image
import os
import win32gui, win32ui
from pathlib import Path

class WindowCapture:
    width = 0
    height = 0
    hwnd = None
    cropped_x = None
    cropped_y = None
    mon = None

    def __init__(self, window_name):
        ctypes.windll.user32.SetProcessDPIAware()
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception("Window not found: {}".format(window_name))
        
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.width = window_rect[2] - window_rect[0]
        self.height = window_rect[3] - window_rect[1]

    def get_screenshot(self, x_0_crop=None, y_0_crop=None, x_1_crop=None, y_1_crop=None):
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.width, self.height)
        saveDC.SelectObject(saveBitMap)

        # Capture the window
        result = ctypes.windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)
        if result:
            file_path = os.path.join(Path.cwd().parent, "images\window_capture.bmp")
            saveBitMap.SaveBitmapFile(saveDC, file_path)
            image = Image.open(file_path)
            if x_0_crop == None:
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwndDC)
                win32gui.DeleteObject(saveBitMap.GetHandle())
                return image
            else: 
                cropped_image = image.crop((x_0_crop, y_0_crop, x_1_crop, y_1_crop))
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwndDC)
                win32gui.DeleteObject(saveBitMap.GetHandle())
                return cropped_image
        else: 
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            win32gui.DeleteObject(saveBitMap.GetHandle())
            print("Failed to capture the window!")
            return None
        
    def get_window_resolution(self):
        return "1920x1080"
    
    def get_window(self):
        return self.hwnd

    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)
