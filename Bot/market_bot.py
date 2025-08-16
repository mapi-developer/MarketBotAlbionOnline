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
    google_sheet = None
    mouse_targets = None
    screenshot_positions = None
    prices_caerleon_local_sheet_file_path = None
    prices_caerleon_local = None
    current_game_frame = None

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        credentials_dir = os.path.join(BASE_DIR, 'items-prices-albion-credentials.json')  
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_dir, configuration.google_sheet_scope)
        client = gspread.authorize(credentials)
        self.google_sheet = client.open(configuration.google_sheet_name)

        self.window_capture = window_capture.WindowCapture(BASE_DIR, configuration.window_title)
        window_resolution = self.window_capture.get_window_resolution()
        self.mouse_targets = configuration.mouse_targets[window_resolution]
        self.screenshot_positions = configuration.screenshot_positions[window_resolution]

        self.prices_caerleon_local_sheet_file_path = os.path.join(BASE_DIR, "tables\prices_caerleon.csv")
        self.prices_caerleon_local = pandas.read_csv(self.prices_caerleon_local_sheet_file_path)

        self.current_game_frame = ""
        print("Bot Initialized")


    def get_current_game_frame(self):
        for key in configuration.game_frames:
            if self.window_capture.get_text_from_screenshot(self.screenshot_positions[key]).replace(" ", "") == configuration.game_frames[key]:
                return configuration.game_frames[key]
        
        return ""

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
                while self.current_game_frame != "gamesettings":
                    pyautogui.press("esc")
                    self.current_game_frame = self.get_current_game_frame()
                    time.sleep(.1)
            
                pyautogui.click(self.window_capture.get_text_screen_position("logout"))
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


    # market functions
    def check_item_price(self, item_full_title, city_name):
        print(item_full_title)
        item_name, item_tier, item_enchantment = item_full_title.split("_")
        best_item_price = 0

        pyautogui.click(self.mouse_targets["market_search_reset"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["market_search"])
        pyautogui.typewrite(item_name)
        pyautogui.click(self.mouse_targets["market_tier"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets[f"market_tier_{item_tier}"])
        pyautogui.click(self.mouse_targets["market_enchantment"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets[f"market_enchantment_{item_enchantment}"])

        for i in configuration.qualities_list:
            pyautogui.click(self.mouse_targets["market_quality"])
            time.sleep(.1)
            pyautogui.click(self.mouse_targets[f"market_quality_{i}"])
            time.sleep(.1)

            text = self.window_capture.get_text_from_screenshot(self.screenshot_positions[f"sell_price_{city_name}"])
            if text != "":
                try:
                    item_price = int("".join(filter(str.isdigit, text)))
                    if item_price > best_item_price:
                        best_item_price = item_price  
                except ValueError:
                    print("Value Error with item price check")       

        return best_item_price

    def update_items_price(self, data_frame, city_name, only_zero_price=False, items_categories_to_check=["all"]):
        items_categories_to_check=["cowl"]
        pyautogui.click(self.mouse_targets["market_buy_tab"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["price_sort_button"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["price_highest_button"])

        for column_index in range(1, len(data_frame.columns), 2):
            if items_categories_to_check[0] == "all" or data_frame.columns[column_index] in items_categories_to_check:
                for row_index in range(0, len(data_frame.iloc[:, column_index])):
                    if data_frame.iloc[row_index, column_index] != "" and type(data_frame.iloc[row_index, column_index]) != type(1.11):
                        item_full_title = data_frame.iloc[row_index, column_index]
                        if only_zero_price == False or data_frame.iloc[row_index, column_index+1] == str(0):
                            price = self.check_item_price(item_full_title, "caerleon")
                            print(price)
                            data_frame.iloc[row_index, column_index+1] = self.check_item_price(item_full_title, "caerleon")

        data_frame.iloc[0, 0] = datetime.strptime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

        data_frame.to_csv(self.prices_caerleon_local_sheet_file_path, index=False)
        worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[city_name])
        local_data = open(self.prices_caerleon_local_sheet_file_path, "r")
        values = [r for r in csv.reader(local_data)]
        worksheet.update(values)
        print("Updated prices from game")

    def check_prices_date(self, city_name, only_zero_price=False):        
        worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[city_name])
        google_sheet_data = worksheet.get_all_values()
        google_sheet_data_frame = pandas.DataFrame(google_sheet_data[1:], columns=google_sheet_data[0])
        self.prices_caerleon_local = pandas.read_csv(self.prices_caerleon_local_sheet_file_path)
        google_sheet_last_update = google_sheet_data_frame.iloc[0]["last update"]
        local_sheet_last_update = self.prices_caerleon_local.iloc[0]["last update"]
        google_sheet_last_update = datetime.strptime(google_sheet_last_update, '%Y-%m-%d %H:%M:%S')
        local_sheet_last_update = datetime.strptime(local_sheet_last_update, '%Y-%m-%d %H:%M:%S')
        current_datetime = datetime.strptime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

        if local_sheet_last_update >= google_sheet_last_update:
            if int((current_datetime - local_sheet_last_update).total_seconds()) > configuration.prices_update_time_gap:
                self.update_items_price(self.prices_caerleon_local, city_name, only_zero_price)
            else:
                local_data = open(self.prices_caerleon_local_sheet_file_path, "r")
                values = [r for r in csv.reader(local_data)]
                worksheet.update(values)
                print("Updated prices in DataBase from local data")
        elif local_sheet_last_update < google_sheet_last_update:
            if int((current_datetime - google_sheet_last_update).total_seconds()) > configuration.prices_update_time_gap:
                self.update_items_price(self.prices_caerleon_local, city_name, only_zero_price)
            else:
                google_sheet_data_frame.to_csv(self.prices_caerleon_local_sheet_file_path, index=False)
                print("Updated prices in local data from DataBase")


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
            #self.check_mouse_click_position()
            self.window_capture.get_text_from_screenshot(self.screenshot_positions[f"sell_price_caerleon"])
        else:
            self.window_capture.set_foreground_window()
            #print(self.get_current_game_frame())
            #print(self.get_account_silver_balance())
            #self.change_account('main_account', 2)
            #self.check_prices_date('caerleon')
            self.update_items_price(self.prices_caerleon_local, 'caerleon')
