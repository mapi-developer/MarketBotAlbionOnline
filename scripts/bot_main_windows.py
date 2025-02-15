import cv2 as cv
import platform
import pytesseract
import pyautogui
import configuration
import time
import os
import re
import gspread
import pandas
import csv
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from datetime import datetime, timezone
import win32con
import win32gui
import windowcapture_windows as windowcapture
local_sheet_file_path = os.path.join(Path.cwd().parent, "data\PricesCaerleon.csv")

credentials = ServiceAccountCredentials.from_json_keyfile_name('items-prices-albion-credentials.json', configuration.google_sheet_scope)
client = gspread.authorize(credentials)
spreadsheet = client.open("MarketBotAlbion")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
current_platform = platform.system()
if current_platform == "Windows":
    print("ok")
elif current_platform == "Darwin":
    import windowcapture_darwin as windowcapture
    local_sheet_file_path = os.path.join(Path.cwd().parent, "data/PricesCaerleon.csv")
window_capture = windowcapture.WindowCapture(configuration.window_title)
window_resolution = window_capture.get_window_resolution()
if window_resolution == "1920x1080":
    mouse_targets = configuration.mouse_targets_1920x1080
    screenshot_positions = configuration.screenshots_positions_1920x1080
elif window_resolution == "1440x900":
    mouse_targets = configuration.mouse_targets_1440x900
    screenshot_positions = configuration.screenshots_positions_1440x900