import ctypes
from PIL import Image
import os
import win32gui, win32ui, win32con
import cv2 as cv
import pytesseract

class WindowCapture:
    BASE_DIR = None
    width = 0
    height = 0
    hwnd = None

    def __init__(self, BASE_DIR, window_name):
        self.BASE_DIR = BASE_DIR
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
            file_path = os.path.join(self.BASE_DIR, "media\window_capture.bmp")
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
        
    def get_text_from_screenshot(self, crop_screenshot_positions, is_gray_reading=True):
        screenshot_file_path = os.path.join(self.BASE_DIR, "media\cropped_zone_screenshot.png")
        gray_screenshot_file_path = os.path.join(self.BASE_DIR, "media\gray_cropped_zone_screenshot.png")
        screenshot = self.get_screenshot(crop_screenshot_positions[0], crop_screenshot_positions[1], crop_screenshot_positions[2], crop_screenshot_positions[3])
        screenshot.save(screenshot_file_path)
        screenshot = cv.imread(screenshot_file_path)
        if is_gray_reading == True:
            gray_screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
            gray_screenshot = cv.threshold(gray_screenshot, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            cv.imwrite(gray_screenshot_file_path, gray_screenshot)
            text = pytesseract.image_to_string(gray_screenshot, config="--psm 6").rstrip("\n").lower()
        else:
            text = pytesseract.image_to_string(screenshot, config="--psm 6").rstrip("\n").lower()
        return text
        
    def get_window_resolution(self):
        return f"{self.width}x{self.height}"
    
    def get_window(self):
        return self.hwnd
    
    def set_foreground_window(self):
        win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(self.hwnd)
