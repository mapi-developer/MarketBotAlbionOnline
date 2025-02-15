import cv2 as cv
import platform
import pytesseract
import pyautogui
import configuration as configuration
import time
import os
import re
import gspread
import pandas
import csv
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from datetime import datetime, timezone
from login_data import login_data

credentials = ServiceAccountCredentials.from_json_keyfile_name("items-prices-albion-credentials.json", configuration.google_sheet_scope)
client = gspread.authorize(credentials)
spreadsheet = client.open("MarketBotAlbion")
import windowcapture_darwin as windowcapture
local_sheet_file_path = os.path.join(Path.cwd().parent, "data/PricesCaerleon.csv")
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
window_capture = windowcapture.WindowCapture(configuration.window_title)
window_resolution = window_capture.get_window_resolution()
mouse_targets = configuration.mouse_targets[window_resolution]
screenshot_positions = configuration.screenshot_positions[window_resolution]

local_sheet_data_frame = pandas.read_csv(local_sheet_file_path)

def check_mouse_click_position():
    from pynput.mouse import Listener, Button
    def on_click(x, y, button, pressed):
        if pressed and button == Button.left:
            print(f'x={x} and y={y}')
        if pressed and button == Button.right:
            return False
    with Listener(on_click=on_click) as listener:
        listener.join()

print(window_capture.get_text_from_screenshot(screenshot_positions["check_game_frame_characters"]))
#check_mouse_click_position()