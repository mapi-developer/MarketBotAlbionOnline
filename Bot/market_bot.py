import window_capture
import configuration
from login_data import login_data

from oauth2client.service_account import ServiceAccountCredentials
import pytesseract
import pyautogui
import time
import os, sys
import re
import gspread
import pandas
import csv
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

        self.window_capture = window_capture.WindowCapture(base_dir=BASE_DIR, window_name=configuration.window_title, debugging=False)
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

        while self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_inventory_open"]).replace(" ", "") != "invenfory":
            pyautogui.press("i")
            time.sleep(.3)
        silver_balance_text_inentory = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_account_silver_from_inventory"], True)
        silver_balance_number_inventory = ConvertSilverToNumber(silver_balance_text_inentory)
        silver_balance_text = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_account_silver"], True)
        silver_balance_number = ConvertSilverToNumber(silver_balance_text)

        print(silver_balance_text, silver_balance_text_inentory)

        if silver_balance_number_inventory == False and silver_balance_number != False:
            return silver_balance_number
        elif silver_balance_number_inventory != False and silver_balance_number == False:
            return silver_balance_number_inventory
        elif silver_balance_number_inventory != False and silver_balance_number != False:
            return silver_balance_number_inventory
        else:
            return False

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

    def check_statistics_open(self):
        if self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_buy_orders_title"]) == "buy orders":
            return True
        else:
            pyautogui.click(self.mouse_targets["extend_item_statistic"])
            time.sleep(.2)
            if self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_buy_orders_title"]) == "buy orders":
                return True
            else:
                return self.check_statistics_open(self)

    # market functions
    # price checking
    def check_item_price(self, item_full_title, city_name):
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

    def update_items_price(self, data_frame=None, city_name="caerleon", only_zero_price=False, items_categories_to_check=["all"]):
        if data_frame == None:
            worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[city_name])
            google_sheet_data = worksheet.get_all_values()
            data_frame = pandas.DataFrame(google_sheet_data[1:], columns=google_sheet_data[0])
            
        pyautogui.click(self.mouse_targets["market_buy_tab"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["price_sort_button"])
        time.sleep(.1)
        pyautogui.click(self.mouse_targets["price_highest_button"])
        #time.sleep(.2)
        #pyautogui.click(self.mouse_targets['item_sort_button'])

        for column_index in range(1, len(data_frame.columns), 2):
            if items_categories_to_check[0] == "all" or data_frame.columns[column_index].lower() in items_categories_to_check:
                for row_index in range(0, len(data_frame.iloc[:, column_index])):
                    if data_frame.iloc[row_index, column_index] != "" and type(data_frame.iloc[row_index, column_index]) != type(1):
                        item_full_title = data_frame.iloc[row_index, column_index]
                        if only_zero_price == False or data_frame.iloc[row_index, column_index+1] == str(0) and item_full_title !="":
                            price = str(self.check_item_price(item_full_title, "caerleon"))
                            print(f'{item_full_title} - {price}')
                            data_frame.iloc[row_index, column_index+1] = price

                data_frame.to_csv(self.prices_caerleon_local_sheet_file_path, index=False)
                worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[city_name])
                local_data = open(self.prices_caerleon_local_sheet_file_path, "r")
                values = [r for r in csv.reader(local_data)]
                worksheet.update(values)

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

    # orders
    def cancel_orders(self, market_type="royal_market"):
        orders_tab, check_order, cancel_order = [], [], []
        if market_type == "royal_market":
            orders_tab = self.mouse_targets["my_orders_tab"]
            check_order = self.screenshot_positions["check_order_exist"]
            cancel_order = self.mouse_targets["cancel_order"]
        elif market_type == "black_market":
            orders_tab = self.mouse_targets["my_orders_tab_black_market"]
            check_order = self.screenshot_positions["check_order_exist"]
            cancel_order = self.mouse_targets["cancel_order_black_market"]

        pyautogui.click(orders_tab)
        time.sleep(.2)
        start_time = time.time()
        while self.window_capture.get_text_from_screenshot(check_order) != "":
            if time.time() - start_time >= 5:
                start_time = time.time()
                pyautogui.scroll(10)
            pyautogui.click(cancel_order)
            time.sleep(.1)

        print("Successfully removed all orders")
 
    def make_buy_orders(self, city_name, items_categories_to_check=["all"]):
        silver_balance = self.get_account_silver_balance()
        pyautogui.click(self.mouse_targets["market_create_buy_order_tab"])
        time.sleep(.1)

        worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[f"items_to_buy_{city_name}"])
        items_to_buy_data = worksheet.get_all_values()
        items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])

        prices_worksheet = self.google_sheet.get_worksheet(configuration.database_sheets['caerleon'])
        google_sheet_data = prices_worksheet.get_all_values()
        prices_caerleon_data_frame = pandas.DataFrame(google_sheet_data[1:], columns=google_sheet_data[0])
        
        i = 0
        for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
            if items_categories_to_check == ["all"] or items_to_buy_data_frame.columns[column_index].lower() in items_categories_to_check:
                for row_index in range(0, len(items_to_buy_data_frame.iloc[:, column_index])):
                    if items_to_buy_data_frame.iloc[row_index, column_index] != "" and type(items_to_buy_data_frame.iloc[row_index, column_index]) != type(1.11):
                        if silver_balance == False or silver_balance > configuration.minimum_account_silver_balance:
                            i += 1
                            if i%10 == 0:
                                silver_balance = self.get_account_silver_balance()
                            item_category = items_to_buy_data_frame.columns[column_index]
                            item_full_title = items_to_buy_data_frame.iloc[row_index, column_index]
                            item_price_column_index = int(prices_caerleon_data_frame.columns.get_loc(item_category))
                            item_price_row_index = int(prices_caerleon_data_frame.index[prices_caerleon_data_frame[item_category] == item_full_title].tolist()[0])
                            item_price_caerleon = int(prices_caerleon_data_frame.iat[item_price_row_index, item_price_column_index+1])
                            item_name, item_tier, item_enchantment = item_full_title.split("_")
                            pyautogui.click(self.mouse_targets["market_search_reset"])
                            time.sleep(.1)
                            pyautogui.click(self.mouse_targets["market_search"])
                            pyautogui.typewrite(item_name)
                            time.sleep(.1)
                            pyautogui.click(self.mouse_targets["market_tier"])
                            time.sleep(.1)
                            pyautogui.click(self.mouse_targets[f"market_tier_{item_tier}"])
                            pyautogui.click(self.mouse_targets["market_enchantment"])
                            time.sleep(.1)
                            pyautogui.click(self.mouse_targets[f"market_enchantment_{item_enchantment}"])
                            time.sleep(.1)
                            pyautogui.click(self.mouse_targets["buy_order_button"])
                            time.sleep(.1)
                            if i == 1:
                                self.check_statistics_open()

                            text = self.window_capture.get_text_from_screenshot(self.screenshot_positions["buy_order_price"])
                            if text != "":
                                try:
                                    item_order_price = int("".join(filter(str.isdigit, text)))
                                except ValueError:
                                    print("Value Error with item price check")
                            item_amount = configuration.get_items_amount(item_order_price)
                            silver_to_buy = item_amount * item_order_price * 1.05
                            #print(item_price_caerleon, item_order_price, (item_price_caerleon/item_order_price)-1)
                            print(silver_balance)
                            if item_order_price < 200000 and item_order_price * configuration.minimum_order_profit_rate < item_price_caerleon and (silver_balance > silver_to_buy or silver_balance == False):
                                time.sleep(.1)
                                pyautogui.click(self.mouse_targets["change_item_amount_in_order"])
                                pyautogui.typewrite(str(item_amount))
                                time.sleep(.1)
                                pyautogui.click(self.mouse_targets["one_silver_more"])
                                pyautogui.click(self.mouse_targets["create_order_button"])
                                time.sleep(.1)
                                pyautogui.click(self.mouse_targets["crate_order_confirmation"])
                                time.sleep(.1)
                                print(f"Made order on {item_name} {item_tier}.{item_enchantment}")
                            else:
                                pyautogui.click(self.mouse_targets["close_order_tab"])
                                time.sleep(.1)
                        else:
                            print("Not enough silver to continue")
                            return False

    # fast buy
    def buy_item_from_market(self, item_caerleon_price, item_full_title, item_bought=0):
        if item_bought >= 5:
            return True
        item_name, item_tier, item_enchantment = item_full_title.split("_")
        item_price = 0
        have_enough_silver = True
        canMakeOrder = False
        avaliable_amount = 1

        text = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_item_price_royal_city"])
        if text != "":
            try:
                item_price = int("".join(filter(str.isdigit, text)))
                if item_price * configuration.minimum_fast_buy_profit_rate < item_caerleon_price:
                    canMakeOrder = True
            except ValueError:
                print("Value Error with item price check")

        if canMakeOrder == True:
            pyautogui.click(self.mouse_targets["buy_order_button"])
            time.sleep(.2)
            self.check_statistics_open()

            avaliable_amount = self.window_capture.get_text_from_screenshot(self.screenshot_positions["check_avaliable_amount"], True)
            try:
                avaliable_amount = int("".join(filter(str.isdigit, avaliable_amount)))
            except ValueError:
                avaliable_amount = 1

            silver_to_buy = item_price * avaliable_amount * 1.05
            silver_on_account = self.get_account_silver_balance()
            if silver_on_account == "can't check silver":
                print("Can't check silver")
            elif silver_on_account < silver_to_buy:
                have_enough_silver = False

        if canMakeOrder == True and have_enough_silver == True:
            time.sleep(.1)
            if avaliable_amount > 1:
                pyautogui.click(self.mouse_targets["change_item_amount_in_order"])
                if avaliable_amount > 10:
                    pyautogui.typewrite(str(10))
                else:
                    pyautogui.typewrite(str(avaliable_amount))
                item_bought += avaliable_amount
            else:
                item_bought += 1
            pyautogui.click(self.mouse_targets["create_order_button"])
            time.sleep(.5)
            print(f"Successfully bought {item_name} {item_tier}.{item_enchantment} - ({avaliable_amount}) || Profit - {item_caerleon_price - item_price}")
            return self.buy_item_from_market(item_caerleon_price, item_full_title, item_bought=item_bought)
        else:
            if have_enough_silver == False:
                return False
            pyautogui.click(self.mouse_targets["close_order_tab"])
            time.sleep(.1)
            print(f"Can't fast buy {item_name} {item_tier}.{item_enchantment}")
            return True

    def fast_buy_items(self, items_categories_to_check=['all']):
        silver_balance = self.get_account_silver_balance()
        pyautogui.click(self.mouse_targets["market_buy_tab"])
        time.sleep(.2)
        pyautogui.click(self.mouse_targets["price_sort_button"])
        time.sleep(.2)
        pyautogui.click(self.mouse_targets["price_lowest_button"])

        worksheet = self.google_sheet.get_worksheet(configuration.database_sheets[f"fast_buy_items"])
        
        items_to_buy_data = worksheet.get_all_values()
        items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])

        prices_worksheet = self.google_sheet.get_worksheet(configuration.database_sheets['caerleon'])
        google_sheet_data = prices_worksheet.get_all_values()
        prices_caerleon_data_frame = pandas.DataFrame(google_sheet_data[1:], columns=google_sheet_data[0])

        for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
            if items_categories_to_check[0] == 'all' or items_to_buy_data_frame.columns[column_index].lower() in items_categories_to_check:
                for row_index in range(0, len(items_to_buy_data_frame.iloc[:, column_index])):
                    if items_to_buy_data_frame.iloc[row_index, column_index] != "" and type(items_to_buy_data_frame.iloc[row_index, column_index]) != type(1.11):
                        if silver_balance > configuration.minimum_account_silver_balance:
                            item_category = items_to_buy_data_frame.columns[column_index]
                            item_full_title = items_to_buy_data_frame.iloc[row_index, column_index]
                            item_price_column_index = int(prices_caerleon_data_frame.columns.get_loc(item_category))
                            item_price_row_index = int(prices_caerleon_data_frame.index[prices_caerleon_data_frame[item_category] == item_full_title].tolist()[0])
                            item_price_caerleon = int(prices_caerleon_data_frame.iat[item_price_row_index, item_price_column_index+1])

                            item_name, item_tier, item_enchantment = item_full_title.split("_")

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
                            time.sleep(.6)

                            self.buy_item_from_market(item_price_caerleon, item_full_title)
                            time.sleep(.3)

        time.sleep(.2)
        print("Have done fast buying items")

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

    def test(self, DEBUG=True):
        if DEBUG:
            self.window_capture.set_foreground_window()
            #self.check_mouse_click_position()
            self.make_buy_orders(city_name='fort_sterling', items_categories_to_check=['bag', 'robe', 'helmet', 'armor', 'jacket', 'cowl', 'shapeshifter'])
            #self.cancel_orders()
            print("DEBUG")
        else:
            try:
                self.window_capture.set_foreground_window()
                #print(self.get_current_game_frame())
                #print(self.get_account_silver_balance())
                #self.change_account('main_account', 2)
                #self.check_prices_date('caerleon')
                self.update_items_price(city_name='caerleon', only_zero_price=False, items_categories_to_check=['armor', 'jacket', 'robe'])
            except KeyboardInterrupt:
                print('Forced to stop bot!')
 