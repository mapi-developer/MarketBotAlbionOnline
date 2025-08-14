import window_capture
import configuration
from login_data import login_data

from oauth2client.service_account import ServiceAccountCredentials
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
    googel_sheet = None
    mouse_targets = None
    screenshot_positions = None
    prices_caerleon_local = None
    current_game_frame = None

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        credentials_dir = os.path.join(BASE_DIR, 'items-prices-albion-credentials.json')  
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_dir, configuration.google_sheet_scope)
        client = gspread.authorize(credentials)
        self.googel_sheet = client.open(configuration.google_sheet_name)

        self.window_capture = window_capture.WindowCapture(BASE_DIR, configuration.window_title)
        window_resolution = self.window_capture.get_window_resolution()
        self.mouse_targets = configuration.mouse_targets[window_resolution]
        self.screenshot_positions = configuration.screenshot_positions[window_resolution]

        prices_caerleon_local_sheet_file_path = os.path.join(BASE_DIR, "tables\prices_caerleon.csv")
        self.prices_caerleon_local = pandas.read_csv(prices_caerleon_local_sheet_file_path)

        self.current_game_frame = ""
        print("Bot Initialized")

    def get_current_game_frame(self):
        game_frame = ""
        if self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_game_frame_login"]).replace(" ", "") == "login":
            game_frame = "login"
        elif self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_game_frame_drops_popup"]).replace(" ", "") == "drops":
            game_frame = "drops_popup"
        elif self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_game_frmae_premium_popup"]).replace(" ", "") == "premium":
            game_frame = "premium_popup"
        elif self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_game_frame_activities_popup"]).replace(" ", "") == "activities":
            game_frame = "activities_popup"
        elif self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_game_frame_characters"]).replace(" ", "") == "characters":
            game_frame = "characters"
        elif self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_escape_menu"]).replace(" ", "") == "logout":
            game_frame = "escape_menu"

        return game_frame

    def get_account_silver_balance(self):
        def ConvertSilverToNumber(string):
            string = string.strip().replace(" ", "")
            multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}

            match = re.match(r'([\d\.]+)([kmb]?)$', string)
            if match:
                number, suffix = match.groups()
                return int(float(number) * multipliers.get(suffix, 1))
            else:
                return False

        if self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_inventory_open"]).replace(" ", "") != "inventory":
            pyautogui.press("i")
            time.sleep(.3)
        silver_balance_text_inentory = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_account_silver_from_inventory"], True)
        silver_balance_number_inventory = ConvertSilverToNumber(silver_balance_text_inentory)
        silver_balance_text = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_account_silver"], True)
        silver_balance_number = ConvertSilverToNumber(silver_balance_text)

        if silver_balance_number_inventory == False and silver_balance_number != False:
            return silver_balance_number
        elif silver_balance_number_inventory != False and silver_balance_number == False:
            return silver_balance_number_inventory
        elif silver_balance_number_inventory != False and silver_balance_number != False:
            return silver_balance_number_inventory

    def change_account(self, account_name, characer_number=1):
        def logout():
            self.current_game_frame = self.get_current_game_frame()
            if self.current_game_frame == "login":
                return
            elif self.current_game_frame == "characters":
                pyautogui.press("esc")
                time.sleep(.2)
                pyautogui.click(self.mouse_targets["logout_confirmation"])
            else:
                while self.current_game_frame != "escape_menu":
                    pyautogui.press("esc")
                    self.current_game_frame = self.get_current_game_frame()
                    print(self.current_game_frame)
                    time.sleep(.1)
            
                pyautogui.click(self.mouse_targets["logout"])
                time.sleep(10)

            while self.current_game_frame != "login":
                self.current_game_frame = self.get_current_game_frame()
                time.sleep(.3)
            time.sleep(.1)

        self.current_game_frame = self.get_current_game_frame()
        if self.current_game_frame != "login":
            logout()
        pyautogui.click(self.mouse_targets["login_email"])
        pyautogui.typewrite(login_data[account_name]["email"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["login_password"])
        pyautogui.typewrite(login_data[account_name]["password"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["login_button"])
        time.sleep(2)
        while self.current_game_frame != "characters":
            self.current_game_frame = self.get_current_game_frame()
            time.sleep(.1)
        pyautogui.click(self.mouse_targets[f"character_{characer_number}"])
        pyautogui.click(self.mouse_targets["enter_world_button"])
        time.sleep(5)
        print(f"Successfully logged into {account_name}")

    # debug functions
    def check_mouse_click_position(self):
        def on_press(key):
            try:
                print(f"Key pressed: {key.char} | Keycode: {ord(key.char)}")
            except AttributeError:
                print(f"Special key pressed: {key} | Keycode: {key.value.vk if hasattr(key, 'value') and key.value else 'Unknown'}")

        def on_release(key):
            if key == keyboard.Key.shift_l:
                return False

        def on_click(x, y, button, pressed):
            if pressed and button == mouse.Button.left:
                print(f'Left mouse click: [{x}, {y}]')
            if pressed and button == mouse.Button.right:
                print(f'Right mouse click" [{x}, {y}]')
            if pressed and button == mouse.Button.middle:
                return False
            
        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        mouse_listener = mouse.Listener(on_click=on_click)
        
        keyboard_listener.start()
        mouse_listener.start()

        keyboard_listener.join()
        mouse_listener.join()

    def test(self, DEBUG=False):
        if DEBUG:
            self.check_mouse_click_position()
        else:
            self.window_capture.set_foreground_window()
            #print(self.get_current_game_frame())
            #print(self.get_account_silver_balance())
            self.change_account('main_account')
