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

