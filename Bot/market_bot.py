import window_capture
import configuration
from login_data import login_data

import pytesseract
import pyautogui
import time
import os
import re
import gspread
import pandas
import csv
from pathlib import Path
from datetime import datetime, timezone
from pynput import keyboard, mouse

class bot():
    window_capture = None
    width = 0
    height = 0
    hwnd = None

    def __init__(self):
        self.window_capture = window_capture.WindowCapture(configuration.window_title)

    def test(self):
        print(login_data['001'])