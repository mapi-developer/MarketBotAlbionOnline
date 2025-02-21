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
from pynput import keyboard, mouse
import win32con
import win32gui
import windowcapture_windows as windowcapture
from login_data import login_data

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

credentials = ServiceAccountCredentials.from_json_keyfile_name('items-prices-albion-credentials.json', configuration.google_sheet_scope)
client = gspread.authorize(credentials)
spreadsheet = client.open("MarketBotAlbion")
local_sheet_file_path = os.path.join(Path.cwd().parent, "data\PricesCaerleon.csv")

window_capture = windowcapture.WindowCapture(configuration.window_title)
window_resolution = window_capture.get_window_resolution()
mouse_targets = configuration.mouse_targets[window_resolution]
screenshot_positions = configuration.screenshot_positions[window_resolution]

local_sheet_data_frame = pandas.read_csv(local_sheet_file_path)
current_game_frame = ""

def get_current_game_frame():
    game_frame = ""
    if window_capture.get_text_from_screenshot(screenshot_positions["check_game_frame_login"]).replace(" ", "") == "login":
        game_frame = "login"
    elif window_capture.get_text_from_screenshot(screenshot_positions["check_game_frame_drops_popup"]).replace(" ", "") == "drops":
        game_frame = "drops_popup"
    elif window_capture.get_text_from_screenshot(screenshot_positions["check_game_frame_characters"]).replace(" ", "") == "characters":
        game_frame = "characters"
    elif window_capture.get_text_from_screenshot(screenshot_positions["check_escape_menu"]).replace(" ", "") == "logout":
        game_frame = "escape_menu"

    return game_frame

def get_account_silver_balance():
    def ConvertSilverToNumber(string):
        string = string.strip().replace(" ", "")
        multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}

        match = re.match(r'([\d\.]+)([kmb]?)$', string)
        if match:
            number, suffix = match.groups()
            return int(float(number) * multipliers.get(suffix, 1))
        else:
            return False

    if window_capture.get_text_from_screenshot(screenshot_positions["check_inventory_open"]).replace(" ", "") != "inventory":
        pyautogui.press("t")
        time.sleep(.3)
    silver_balance_text_inentory = window_capture.get_text_from_screenshot(screenshot_positions["check_account_silver_from_inventory"], True)
    silver_balance_number_inventory = ConvertSilverToNumber(silver_balance_text_inentory)
    silver_balance_text = window_capture.get_text_from_screenshot(screenshot_positions["check_account_silver"], True)
    silver_balance_number = ConvertSilverToNumber(silver_balance_text)

    if silver_balance_number_inventory == False and silver_balance_number != False:
        return silver_balance_number
    elif silver_balance_number_inventory != False and silver_balance_number == False:
        return silver_balance_number_inventory
    elif silver_balance_number_inventory != False and silver_balance_number != False:
        return silver_balance_number_inventory

def change_account(account_name, characer_number=1):
    def logout():
        global current_game_frame
        current_game_frame = get_current_game_frame()
        if current_game_frame == "login":
            return
        while current_game_frame != "escape_menu":
            pyautogui.press("esc")
            current_game_frame = get_current_game_frame()
            time.sleep(.1)
        pyautogui.click(mouse_targets["logout"])
        time.sleep(12)
        while current_game_frame != "login":
            current_game_frame = get_current_game_frame()
            time.sleep(.3)
        time.sleep(.1)

    global current_game_frame
    current_game_frame = get_current_game_frame()
    if current_game_frame != "login":
        logout()
    pyautogui.click(mouse_targets["login_email"])
    pyautogui.typewrite(login_data[account_name]["email"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["login_password"])
    pyautogui.typewrite(login_data[account_name]["password"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["login_button"])
    time.sleep(2)
    while current_game_frame != "characters":
        current_game_frame = get_current_game_frame()
        time.sleep(.1)
    pyautogui.click(mouse_targets[f"character_{characer_number}"])
    pyautogui.click(mouse_targets["enter_world_button"])
    time.sleep(5)
    print(f"Successfully logged into {account_name}")

# market functions
def check_item_price(item_full_title, city_name):
    item_name, item_tier, item_enchantment = item_full_title.split("_")
    best_item_price = 0

    pyautogui.click(mouse_targets["market_search_reset"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["market_search"])
    pyautogui.typewrite(item_name)
    pyautogui.click(mouse_targets["market_tier"])
    time.sleep(.1)
    pyautogui.click(mouse_targets[f"market_tier_{item_tier}"])
    pyautogui.click(mouse_targets["market_enchantment"])
    time.sleep(.1)
    pyautogui.click(mouse_targets[f"market_enchantment_{item_enchantment}"])

    for i in configuration.qualities_list:
        pyautogui.click(mouse_targets["market_quality"])
        time.sleep(.1)
        pyautogui.click(mouse_targets[f"market_quality_{i}"])
        time.sleep(.1)

        text = window_capture.get_text_from_screenshot(screenshot_positions[f"sell_price_{city_name}"])
        if text != "":
            try:
                item_price = int("".join(filter(str.isdigit, text)))
                if item_price > best_item_price:
                    best_item_price = item_price  
            except ValueError:
                print("Value Error with item price check")       

    return best_item_price

def update_items_price(data_frame, city_name, only_zero_price=False, items_categories_to_check=["all"]):
    pyautogui.click(mouse_targets["market_buy_tab"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["price_sort_button"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["price_highest_button"])

    for column_index in range(1, len(data_frame.columns), 2):
        if items_categories_to_check[0] == "all" or data_frame.columns[column_index] in items_categories_to_check:
            for row_index in range(1, len(data_frame.iloc[:, column_index])):
                if data_frame.iloc[row_index, column_index] != "" and type(data_frame.iloc[row_index, column_index]) != type(1.11):
                    item_full_title = data_frame.iloc[row_index, column_index]
                    if only_zero_price == False or data_frame.iloc[row_index, column_index+1] == str(0):
                        data_frame.iloc[row_index, column_index+1] = check_item_price(item_full_title, "caerleon")

    data_frame.iloc[0, 0] = datetime.strptime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    data_frame.to_csv(local_sheet_file_path, index=False)
    worksheet = spreadsheet.get_worksheet(configuration.database_sheets[city_name])
    local_data = open(local_sheet_file_path, "r")
    values = [r for r in csv.reader(local_data)]
    worksheet.update(values)
    print("Updated prices from game")

def check_prices_date(city_name, only_zero_price=False):
    global local_sheet_data_frame
    
    worksheet = spreadsheet.get_worksheet(configuration.database_sheets[city_name])
    
    google_sheet_data = worksheet.get_all_values()
    google_sheet_data_frame = pandas.DataFrame(google_sheet_data[1:], columns=google_sheet_data[0])
    local_sheet_data_frame = pandas.read_csv(local_sheet_file_path)
    google_sheet_last_update = google_sheet_data_frame.iloc[0]["last update"]
    local_sheet_last_update = local_sheet_data_frame.iloc[0]["last update"]
    google_sheet_last_update = datetime.strptime(google_sheet_last_update, '%Y-%m-%d %H:%M:%S')
    local_sheet_last_update = datetime.strptime(local_sheet_last_update, '%Y-%m-%d %H:%M:%S')
    current_datetime = datetime.strptime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    if local_sheet_last_update >= google_sheet_last_update:
        if int((current_datetime - local_sheet_last_update).total_seconds()) > configuration.prices_update_time_gap:
            update_items_price(local_sheet_data_frame, city_name, only_zero_price)
        else:
            local_data = open(local_sheet_file_path, "r")
            values = [r for r in csv.reader(local_data)]
            worksheet.update(values)
            print("Updated prices in DataBase from local data")
    elif local_sheet_last_update < google_sheet_last_update:
        if int((current_datetime - google_sheet_last_update).total_seconds()) > configuration.prices_update_time_gap:
            update_items_price(local_sheet_data_frame, city_name, only_zero_price)
        else:
            google_sheet_data_frame.to_csv(local_sheet_file_path, index=False)
            print("Updated prices in local data from DataBase")

def cancel_orders(market_type="royal_market"):
    if market_type == "royal_market":
        pyautogui.click(mouse_targets["my_orders_tab"])
        time.sleep(.2)
        start_time = time.time()
        while window_capture.get_text_from_screenshot(screenshot_positions["check_order_exist"]) != "":
            if time.time() - start_time >= 5:
                start_time = time.time()
                pyautogui.scroll(10)
            pyautogui.click(mouse_targets["cancel_order"])
            time.sleep(.1)

        print("Successfully removed all orders")
    elif market_type == "black_market":
        pyautogui.click(mouse_targets["my_orders_tab_black_market"])
        time.sleep(.2)
        start_time = time.time()
        while window_capture.get_text_from_screenshot(screenshot_positions["check_order_exist"]) != "":
            if time.time() - start_time >= 5:
                start_time = time.time()
                pyautogui.scroll(10)
            pyautogui.click(mouse_targets["cancel_order_black_market"])
            time.sleep(.1)

        print("Successfully removed all orders")

def check_statistics_open():
    if window_capture.get_text_from_screenshot(screenshot_positions["check_buy_orders_title"]) == "buy orders":
        return True
    else:
        pyautogui.click(mouse_targets["extend_item_statistic"])
        time.sleep(.2)

def make_buy_orders(city_name, items_categories_to_check=["all"]):
    silver_balance = get_account_silver_balance()
    pyautogui.click(mouse_targets["market_create_buy_order_tab"])
    time.sleep(.1)

    worksheet = spreadsheet.get_worksheet(configuration.database_sheets[f"items_to_buy_{city_name}"])
    
    items_to_buy_data = worksheet.get_all_values()
    items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])
    prices_caerleon_data_frame = pandas.read_csv(local_sheet_file_path)
    
    i = 0
    for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
        if items_categories_to_check == ["all"] or items_to_buy_data_frame.columns[column_index] in items_categories_to_check:
            for row_index in range(0, len(items_to_buy_data_frame.iloc[:, column_index])):
                if items_to_buy_data_frame.iloc[row_index, column_index] != "" and type(items_to_buy_data_frame.iloc[row_index, column_index]) != type(1.11):
                    if silver_balance > configuration.minimum_account_silver_balance:
                        i += 1
                        if i%10 == 0:
                            silver_balance = get_account_silver_balance()
                        item_category = items_to_buy_data_frame.columns[column_index]
                        item_full_title = items_to_buy_data_frame.iloc[row_index, column_index]
                        item_price_column_index = int(prices_caerleon_data_frame.columns.get_loc(item_category))
                        item_price_row_index = int(prices_caerleon_data_frame.index[prices_caerleon_data_frame[item_category] == item_full_title].tolist()[0])
                        item_price_caerleon = int(prices_caerleon_data_frame.iat[item_price_row_index, item_price_column_index+1])
                        item_name, item_tier, item_enchantment = item_full_title.split("_")
                        pyautogui.click(mouse_targets["market_search_reset"])
                        time.sleep(.1)
                        pyautogui.click(mouse_targets["market_search"])
                        pyautogui.typewrite(item_name)
                        time.sleep(.1)
                        pyautogui.click(mouse_targets["market_tier"])
                        time.sleep(.1)
                        pyautogui.click(mouse_targets[f"market_tier_{item_tier}"])
                        pyautogui.click(mouse_targets["market_enchantment"])
                        time.sleep(.1)
                        pyautogui.click(mouse_targets[f"market_enchantment_{item_enchantment}"])
                        time.sleep(.1)
                        pyautogui.click(mouse_targets["buy_order_button"])
                        time.sleep(.1)
                        if i == 1:
                            check_statistics_open()

                        text = window_capture.get_text_from_screenshot(screenshot_positions["buy_order_price"])
                        if text != "":
                            try:
                                item_order_price = int("".join(filter(str.isdigit, text)))
                            except ValueError:
                                print("Value Error with item price check")
                        item_amount = configuration.get_items_amount(item_order_price)
                        silver_to_buy = item_amount * item_order_price * 1.05
                        print(silver_balance, silver_to_buy)
                        if item_order_price * configuration.minimum_order_profit_rate < item_price_caerleon and silver_balance > silver_to_buy:
                            time.sleep(.1)
                            pyautogui.click(mouse_targets["change_item_amount_in_order"])
                            pyautogui.typewrite(str(item_amount))
                            time.sleep(.1)
                            pyautogui.click(mouse_targets["one_silver_more"])
                            pyautogui.click(mouse_targets["create_order_button"])
                            time.sleep(.1)
                            pyautogui.click(mouse_targets["crate_order_confirmation"])
                            time.sleep(.1)
                            print(f"Made order on {item_name} {item_tier}.{item_enchantment}")
                        else:
                            pyautogui.click(mouse_targets["close_order_tab"])
                            time.sleep(.1)
                    else:
                        print("Not enough silver to continue")
                        return False
                
def BuyItemFromMarket(item_caerleon_price, item_full_title):
    item_name, item_tier, item_enchantment = item_full_title.split("_")
    item_price = 0
    have_enough_silver = True
    canMakeOrder = False
    avaliable_amount = 1

    text = window_capture.get_text_from_screenshot(screenshot_positions["check_item_price_royal_city"])
    if text != "":
        try:
            item_price = int("".join(filter(str.isdigit, text)))
            if item_price * configuration.minimum_fast_buy_profit_rate < item_caerleon_price:
                canMakeOrder = True
        except ValueError:
            print("Value Error with item price check")

    if canMakeOrder == True:
        pyautogui.click(mouse_targets["buy_order_button"])
        time.sleep(.2)
        check_statistics_open()

        avaliable_amount = window_capture.get_text_from_screenshot(screenshot_positions["check_avaliable_amount"], True)
        try:
            avaliable_amount = int("".join(filter(str.isdigit, avaliable_amount)))
        except ValueError:
            avaliable_amount = 1

        silver_to_buy = item_price * avaliable_amount * 1.05
        silver_on_account = get_account_silver_balance()
        if silver_on_account == "can't check silver":
            print("Can't check silver")
        elif silver_on_account < silver_to_buy:
            have_enough_silver = False

    if canMakeOrder == True and have_enough_silver == True:
        time.sleep(.1)
        if avaliable_amount > 1:
            pyautogui.click(mouse_targets["change_item_amount_in_order"])
            pyautogui.typewrite(str(avaliable_amount))
        pyautogui.click(mouse_targets["create_order_button"])
        time.sleep(.2)
        print(f"Successfully bought {item_name} {item_tier}.{item_enchantment}")
        return BuyItemFromMarket(item_caerleon_price, item_full_title)
    else:
        if have_enough_silver == False:
            return False
        pyautogui.click(mouse_targets["close_order_tab"])
        time.sleep(.1)
        print(f"Can't fast buy {item_name} {item_tier}.{item_enchantment}")
        return True

def fast_buy_items():
    silver_balance = get_account_silver_balance()
    pyautogui.click(mouse_targets["market_buy_tab"])
    time.sleep(.1)

    worksheet = spreadsheet.get_worksheet(configuration.database_sheets[f"fast_buy_items"])
    
    items_to_buy_data = worksheet.get_all_values()
    items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])
    prices_caerleon_data_frame = pandas.read_csv(local_sheet_file_path)
    
    i = 0
    for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
        for row_index in range(0, len(items_to_buy_data_frame.iloc[:, column_index])):
            if items_to_buy_data_frame.iloc[row_index, column_index] != "" and type(items_to_buy_data_frame.iloc[row_index, column_index]) != type(1.11):
                if silver_balance > configuration.minimum_account_silver_balance:
                    item_category = items_to_buy_data_frame.columns[column_index]
                    item_full_title = items_to_buy_data_frame.iloc[row_index, column_index]
                    item_price_column_index = int(prices_caerleon_data_frame.columns.get_loc(item_category))
                    item_price_row_index = int(prices_caerleon_data_frame.index[prices_caerleon_data_frame[item_category] == item_full_title].tolist()[0])
                    item_price_caerleon = int(prices_caerleon_data_frame.iat[item_price_row_index, item_price_column_index+1])

                    item_name, item_tier, item_enchantment = item_full_title.split("_")

                    pyautogui.click(mouse_targets["market_search_reset"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets["market_search"])
                    pyautogui.typewrite(item_name)
                    pyautogui.click(mouse_targets["market_tier"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets[f"market_tier_{item_tier}"])
                    pyautogui.click(mouse_targets["market_enchantment"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets[f"market_enchantment_{item_enchantment}"])
                    time.sleep(.6)

                    BuyItemFromMarket(item_price_caerleon, item_full_title)
                    time.sleep(.3)

    time.sleep(.2)
    print("Have done fast buying items")

def make_sell_order_on_item():
    pyautogui.click(mouse_targets["buy_order_button"])
    time.sleep(.2)

    item_order_price, sell_item_price = 0, 0

    text = window_capture.get_text_from_screenshot(screenshot_positions["buy_order_price"], False)
    if text != "":
        if "one" in text:
            item_order_price = 0
        else:
            item_order_price = int("".join(filter(str.isdigit, text)))
    
    text = window_capture.get_text_from_screenshot(screenshot_positions["sell_order_price"], False)
    if text != "":
        try:
            sell_item_price = int("".join(filter(str.isdigit, text)))
        except ValueError:
            print("Value Error with item price check")

    price_gap = sell_item_price * configuration.blackmarket_fast_sell_price_gap
    if sell_item_price - price_gap <= item_order_price:
        pyautogui.click(mouse_targets["sell_fast"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["create_order_button"])
    else:
        pyautogui.click(mouse_targets["sell_long"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["one_silver_less"])
        pyautogui.click(mouse_targets["create_order_button"])

    time.sleep(.1)

def make_sell_orders():
    pyautogui.click(mouse_targets["market_sell_tab"])
    time.sleep(.1)

    text = window_capture.get_text_from_screenshot(screenshot_positions["check_items_in_inventory"]) 
    if text == "":
        if window_capture.get_text_from_screenshot(screenshot_positions["order_no_longer_exist_check"])  == "someone":
            pyautogui.click(mouse_targets["order_no_longer_exist_ok"])
            time.sleep(.1)
            pyautogui.click(mouse_targets["market_buy_tab"])
            time.sleep(.1)
            pyautogui.click(mouse_targets["market_sell_tab"])
            time.sleep(.1)
            make_sell_order_on_item()
            return make_sell_orders()
        else:
            print("Selling orders done successfully")
            return True
    else:
        make_sell_order_on_item()
        return make_sell_orders()

def make_market_only_orders():
    pyautogui.click(mouse_targets["market_create_buy_order_tab"])
    time.sleep(.1)

    worksheet = spreadsheet.get_worksheet(configuration.database_sheets["items_to_buy_market_only"])
    items_to_buy_data = worksheet.get_all_values()
    items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])

    silver_balance = get_account_silver_balance()

    i = 0
    for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
        for row_index in range(0, len(items_to_buy_data_frame.iloc[:, column_index])):
            if items_to_buy_data_frame.iloc[row_index, column_index] != "" and type(items_to_buy_data_frame.iloc[row_index, column_index]) != type(1.11):
                if silver_balance > configuration.minimum_account_silver_balance:
                    i += 1
                    if i%10 == 0:
                        silver_balance = get_account_silver_balance()
                    item_order_price = 0
                    item_sell_price = 0

                    item_full_title = items_to_buy_data_frame.iloc[row_index, column_index]
                    item_name, item_tier, item_enchantment = item_full_title.split("_")
                    pyautogui.click(mouse_targets["market_search_reset"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets["market_search"])
                    pyautogui.typewrite(item_name)
                    time.sleep(.1)
                    pyautogui.click(mouse_targets["market_tier"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets[f"market_tier_{item_tier}"])
                    pyautogui.click(mouse_targets["market_enchantment"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets[f"market_enchantment_{item_enchantment}"])
                    time.sleep(.1)
                    pyautogui.click(mouse_targets["buy_order_button"])
                    time.sleep(.1)

                    if i == 1:
                        check_statistics_open()

                    text = window_capture.get_text_from_screenshot(screenshot_positions["buy_order_price"])
                    if text != "":
                        try:
                            item_order_price = int("".join(filter(str.isdigit, text)))
                        except ValueError:
                            print("Value Error with item price check")
                    text = window_capture.get_text_from_screenshot(screenshot_positions["sell_order_price"])
                    if text != "":
                        try:
                            item_sell_price = int("".join(filter(str.isdigit, text)))
                        except ValueError:
                            print("Value Error with item price check")

                    if item_sell_price != 0 and item_order_price != 0:
                        if item_order_price * configuration.minimum_market_only_profit_rate <= item_sell_price:
                            pyautogui.click(mouse_targets["one_silver_more"])
                            pyautogui.click(mouse_targets["create_order_button"])
                            time.sleep(.1)
                            pyautogui.click(mouse_targets["crate_order_confirmation"])
                            time.sleep(.1)
                        else:
                            pyautogui.click(mouse_targets["close_order_tab"])
                            time.sleep(.2)   

                        print(item_full_title, item_sell_price, item_order_price)
                    else:
                        pyautogui.click(mouse_targets["close_order_tab"])
                        time.sleep(.2)

# islands (farm, workers) functions
def travel_to_island(island_name="Matvey4a Guild's Island - Fort Sterling"):
    pyautogui.click(mouse_targets["travel_to"])
    time.sleep(.1)
    pyautogui.typewrite(island_name)
    time.sleep(.1)
    pyautogui.moveTo(mouse_targets["travel_to_drop_down"], duration=.1)
    pyautogui.click(mouse_targets["travel_to_drop_down"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["buy_journey_button"])
    time.sleep(8)

def plant_island(plot_type="farm"):
    def plant_farm_spot():
        pyautogui.press("t")
        time.sleep(.1)
        pyautogui.click(1599, 954)
        time.sleep(.1)
        pyautogui.click(1558, 953)
        time.sleep(.2)
        pyautogui.click(1533, 557)
        time.sleep(.1)
        pyautogui.click(746, 371)
        time.sleep(1)
        pyautogui.press("t")
        pyautogui.press("a")
        time.sleep(1)
        pyautogui.moveTo(884, 301, .2)
        pyautogui.click(884, 301)
        time.sleep(2)
        pyautogui.moveTo(1046, 255, .2)
        pyautogui.click(1046, 255)
        time.sleep(2)
        pyautogui.moveTo(1231, 188, .2)
        pyautogui.click(1231, 188)
        time.sleep(2)
        pyautogui.moveTo(1108, 74, .2)
        pyautogui.click(1108, 74)
        time.sleep(2)
        pyautogui.moveTo(961, 15, .2)
        pyautogui.click(961, 15)
        time.sleep(2)
        pyautogui.moveTo(971, 138, .2)
        pyautogui.click(971, 138)
        time.sleep(2)
        pyautogui.moveTo(807, 61, .2)
        pyautogui.click(807, 61)
        time.sleep(2)
        pyautogui.moveTo(784, 194, .2)
        pyautogui.click(784, 194)
        time.sleep(2)
        pyautogui.moveTo(643, 108, .2)
        pyautogui.click(643, 108)
        time.sleep(2)
        pyautogui.click(1196, 763)
        pyautogui.press("a")
        time.sleep(5.2)
    
    if plot_type == "farm":
        pyautogui.rightClick(1426, 66)
        time.sleep(2.5)
        pyautogui.rightClick(1067, 217)
        time.sleep(1)
        pyautogui.rightClick(198, 580)
        time.sleep(.1)
        plant_farm_spot()
        print("Spot 1 is planted")

        pyautogui.rightClick(1171, 21)
        time.sleep(2.5)
        pyautogui.rightClick(1005, 434)
        time.sleep(.1)
        plant_farm_spot()
        print("Spot 2 is planted")  

        pyautogui.rightClick(1069, 9)
        time.sleep(3)
        pyautogui.rightClick(1015, 18)
        time.sleep(3)
        pyautogui.rightClick(248, 236)
        time.sleep(.1)
        plant_farm_spot()
        print("Spot 3 is planted")  

        pyautogui.rightClick(503, 1079)
        time.sleep(3)
        pyautogui.rightClick(995, 702)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 4 is planted")

        pyautogui.rightClick(1575, 82)
        time.sleep(4)
        pyautogui.rightClick(1811, 755)
        time.sleep(2.5)
        pyautogui.rightClick(1361, 938)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 5 is planted")

        pyautogui.rightClick(1763, 518)
        time.sleep(4)
        pyautogui.rightClick(1442, 1079)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 6 is planted")

        pyautogui.rightClick(1350, 703)
        time.sleep(2)
        pyautogui.rightClick(1171, 851)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 7 is planted")

        pyautogui.rightClick(1501, 132)
        time.sleep(4)
        plant_farm_spot()
        print("Spot 8 is planted")

        pyautogui.rightClick(1337, 722)
        time.sleep(2)
        pyautogui.rightClick(1167, 804)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 9 is planted")

        pyautogui.rightClick(259, 1068)
        time.sleep(2)
        pyautogui.rightClick(692, 873)
        time.sleep(2)
        pyautogui.rightClick(724, 884)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 10 is planted")

        pyautogui.rightClick(758, 594)
        time.sleep(2)
        pyautogui.rightClick(1534, 942)
        time.sleep(2)
        pyautogui.rightClick(1635, 747)
        time.sleep(2)
        pyautogui.rightClick(1413, 719)
        time.sleep(2)
        pyautogui.rightClick(1059, 833)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 11 is planted")

        pyautogui.rightClick(429, 958)
        time.sleep(2)
        pyautogui.rightClick(920, 910)
        time.sleep(2)
        pyautogui.rightClick(1068, 982)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 12 is planted")

        pyautogui.rightClick(554, 31)
        time.sleep(3)
        pyautogui.rightClick(671, 363)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 13 is planted")

        pyautogui.rightClick(1201, 24)
        time.sleep(4)
        pyautogui.rightClick(673, 32)
        time.sleep(3)
        pyautogui.rightClick(616, 224)
        time.sleep(2)
        pyautogui.rightClick(523, 933)
        time.sleep(2)
        pyautogui.rightClick(650, 600)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 14 is planted")

        pyautogui.rightClick(392, 285)
        time.sleep(3)
        pyautogui.rightClick(407, 614)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 15 is planted")

        pyautogui.rightClick(1330, 36)
        time.sleep(4)
        pyautogui.rightClick(1439, 91)
        time.sleep(2)
        pyautogui.rightClick(494, 167)
        time.sleep(2)
        plant_farm_spot()
        print("Spot 16 is planted")

        pyautogui.rightClick(539, 339)
        time.sleep(2)
        pyautogui.rightClick(367, 903)
        time.sleep(2)
        pyautogui.rightClick(320, 906)
        time.sleep(2)
        pyautogui.rightClick(584, 772)
        time.sleep(2)
        pyautogui.click(721, 362)
        time.sleep(.5)

def setup_for_island_planting(status="start", island_name="Matvey4a's Island"):
    pyautogui.click(mouse_targets["go_to_chest_guild_island"])
    time.sleep(5)
    
    if status == "finish":
        pyautogui.click(mouse_targets["bank_choose_tab"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_1_loot_tab"])
        time.sleep(.2)
        pyautogui.click(mouse_targets["bank_move_all_button"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_move_all_button"])
        time.sleep(.2)
        pyautogui.click(mouse_targets["bank_move_all_from_inventory"])
        time.sleep(.5)
        pyautogui.press("esc")
        time.sleep(.1)
        pyautogui.click(mouse_targets["go_to_traveler_from_chest_guild_island"])
        time.sleep(4)
        pyautogui.click(mouse_targets["enter_traveler_from_guild_island"])
        time.sleep(1)
        travel_to_island(island_name)
        time.sleep(8)
        pyautogui.click(813, 477)
        time.sleep(.5)
        travel_to_island(configuration.islands[f"market_guild_fort_sterling"])
        return
    elif status == "start":
        pyautogui.click(mouse_targets["bank_choose_tab"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_2_loot_tab"])
        time.sleep(.2)
        pyautogui.click(mouse_targets["bank_move_all_button"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_move_all_button"])
        time.sleep(.2)
        pyautogui.click(mouse_targets["bank_move_all_from_inventory"])
        time.sleep(.5)

        pyautogui.click(mouse_targets["bank_choose_tab"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_1_loot_tab"])
        time.sleep(.2)
        pyautogui.click(mouse_targets["bank_stack_items"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_sort_items"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_item_1"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_take_item_button"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["inventory_item_1"])
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_take_item_change_amount"])
        time.sleep(.1)
        pyautogui.typewrite(str(144))
        time.sleep(.1)
        pyautogui.click(mouse_targets["bank_take_item_split_button"])
        time.sleep(.2)
        pyautogui.moveTo(mouse_targets["inventory_item_1"])
        pyautogui.mouseDown()
        pyautogui.moveTo(mouse_targets["bank_item_1"], duration=.2)
        pyautogui.mouseUp()
        time.sleep(.3)
        pyautogui.press("esc")
        time.sleep(.1)
        pyautogui.click(mouse_targets["go_to_traveler_from_chest_guild_island"])
        time.sleep(4)
        pyautogui.click(mouse_targets["enter_traveler_from_guild_island"])
        time.sleep(1)

def harvest_island(plot_type="farm"):
    def harvest_farm_spot():
        pyautogui.keyDown("shift")
        pyautogui.click(917, 58)
        time.sleep(2)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(902, 537)
        time.sleep(.1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(765, 356)
        time.sleep(.1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(1092, 385)
        time.sleep(.1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(624, 272)
        time.sleep(1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(970, 261)
        time.sleep(1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(1277, 297)
        time.sleep(1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(991, 282)
        time.sleep(1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.click(1331, 612)
        time.sleep(1)
        #pyautogui.click(1207, 564)
        time.sleep(.1)
        pyautogui.keyUp("shift")
        #pyautogui.press("a")
        time.sleep(.2)
        pyautogui.press("a")
        time.sleep(5.2)

    if plot_type == "farm":
        pyautogui.rightClick(1426, 66)
        time.sleep(2.5)
        pyautogui.rightClick(1067, 217)
        time.sleep(1)
        pyautogui.rightClick(198, 580)
        time.sleep(2.5)
        harvest_farm_spot()
        print("Spot 1 is planted")
        
        pyautogui.rightClick(1154, 294)
        time.sleep(2.5)
        harvest_farm_spot()
        print("Spot 2 is planted")  
        
        pyautogui.rightClick(1135, 32)
        time.sleep(2.5)
        pyautogui.rightClick(552, 236)
        time.sleep(2.5)
        pyautogui.rightClick(591, 343)
        time.sleep(1.5)
        harvest_farm_spot()
        print("Spot 3 is planted")  
        
        pyautogui.rightClick(840, 997)
        time.sleep(2.5)
        pyautogui.rightClick(847, 1028)
        time.sleep(2.5)
        pyautogui.rightClick(636, 840)
        time.sleep(2.5)
        harvest_farm_spot()
        print("Spot 4 is planted")
        
        pyautogui.rightClick(1675, 427)
        time.sleep(4)
        pyautogui.rightClick(1784, 599)
        time.sleep(3.5)
        pyautogui.rightClick(1255, 1049)
        time.sleep(2.5)
        harvest_farm_spot()
        print("Spot 5 is planted")
        
        pyautogui.rightClick(1443, 1016)
        time.sleep(4)
        pyautogui.rightClick(1617, 829)
        time.sleep(3)
        pyautogui.rightClick(1060, 830)
        time.sleep(1.5)
        harvest_farm_spot()
        print("Spot 6 is planted")
        
        pyautogui.rightClick(1178, 1036)
        time.sleep(3)
        pyautogui.rightClick(1344, 969)
        time.sleep(3)
        pyautogui.rightClick(909, 645)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 7 is planted")
        
        pyautogui.rightClick(1495, 571)
        time.sleep(3)
        harvest_farm_spot()
        print("Spot 8 is planted")
        
        pyautogui.rightClick(1110, 1032)
        time.sleep(3)
        pyautogui.rightClick(1348, 817)
        time.sleep(2)
        pyautogui.rightClick(960, 770)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 9 is planted")
        
        pyautogui.rightClick(408, 997)
        time.sleep(3)
        pyautogui.rightClick(751, 1026)
        time.sleep(2)
        pyautogui.rightClick(540, 971)
        time.sleep(2)
        pyautogui.rightClick(736, 982)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 10 is planted")
        
        pyautogui.rightClick(525, 970)
        time.sleep(2.5)
        pyautogui.rightClick(1099, 987)
        time.sleep(2)
        pyautogui.rightClick(1431, 811)
        time.sleep(2)
        pyautogui.rightClick(1744, 672)
        time.sleep(2)
        pyautogui.rightClick(1341, 794)
        time.sleep(2)
        pyautogui.rightClick(988, 771)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 11 is planted")
        
        pyautogui.rightClick(574, 1014)
        time.sleep(3)
        pyautogui.rightClick(826, 992)
        time.sleep(2)
        pyautogui.rightClick(967, 977)
        time.sleep(2)
        pyautogui.rightClick(971, 787)
        time.sleep(2)
        pyautogui.rightClick(797, 670)
        time.sleep(1.5)
        harvest_farm_spot()
        print("Spot 12 is planted")
        
        pyautogui.rightClick(351, 128)
        time.sleep(4)
        pyautogui.rightClick(761, 687)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 13 is planted")
        
        pyautogui.rightClick(970, 59)
        time.sleep(4)
        pyautogui.rightClick(534, 110)
        time.sleep(4)
        pyautogui.rightClick(258, 857)
        time.sleep(3)
        pyautogui.rightClick(861, 711)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 14 is planted") #error
        
        pyautogui.rightClick(191, 494)
        time.sleep(4)
        pyautogui.rightClick(440, 730)
        time.sleep(3)
        pyautogui.rightClick(931, 668)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 15 is planted")
        
        pyautogui.rightClick(1646, 103)
        time.sleep(4)
        pyautogui.rightClick(881, 15)
        time.sleep(2)
        pyautogui.rightClick(687, 578)
        time.sleep(2)
        harvest_farm_spot()
        print("Spot 16 is planted")
        
        pyautogui.rightClick(324, 679)
        time.sleep(3)
        pyautogui.rightClick(658, 915)
        time.sleep(2)
        pyautogui.rightClick(284, 721)
        time.sleep(3)
        pyautogui.rightClick(325, 1047)
        time.sleep(4)
        pyautogui.click(752, 463)
        time.sleep(2)
        pyautogui.click(868, 538)
        time.sleep(.5)

def make_island_cycle(city_name="fort_sterling", island_name="Matvey4a's Island", island_type="player", island_plots_type="farm"):
    if island_type == "player":
        #pyautogui.click(mouse_targets["enter_traveler_after_login"])
        #time.sleep(2)
        #travel_to_island(island_name)
        #harvest_island(island_plots_type)
        travel_to_island(configuration.islands[f"market_guild_{city_name}"])
        setup_for_island_planting()
        travel_to_island(island_name)
        plant_island(island_plots_type)
        travel_to_island(configuration.islands[f"market_guild_{city_name}"])

def workers_cycle():
    def load_journal():
        pyautogui.moveTo(mouse_targets["inventory_item_2"], duration=.2)
        pyautogui.mouseDown()
        pyautogui.moveTo(mouse_targets["worker_journal_slot"], duration=.2)
        pyautogui.mouseUp()
        time.sleep(.2)
        pyautogui.click(mouse_targets["worker_accept_journal_button"])
        time.sleep(.2)

    
    pyautogui.click(mouse_targets["go_to_chest_guild_island"])
    time.sleep(5)

    pyautogui.click(mouse_targets["bank_stack_items"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_sort_items"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_item_1"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_take_item_button"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["inventory_item_1"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_take_item_change_amount"])
    time.sleep(.1)
    pyautogui.typewrite(str(15))
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_take_item_split_button"])
    time.sleep(.2)
    pyautogui.moveTo(mouse_targets["inventory_item_1"])
    pyautogui.mouseDown()
    pyautogui.moveTo(mouse_targets["bank_item_1"], duration=.2)
    pyautogui.mouseUp()
    time.sleep(.3)
    pyautogui.press("esc")
    pyautogui.click(1007, 158)
    time.sleep(2)
    pyautogui.click(1818, 237)
    time.sleep(4)
    pyautogui.click(1175, 185)
    time.sleep(2)
    pyautogui.click(1003, 322)
    time.sleep(1)
    pyautogui.click(662, 465)
    time.sleep(1)
    load_journal()
    print("journal_1_loaded")
    pyautogui.click(926, 381)
    time.sleep(.2)
    load_journal()
    print("journal_2_loaded")
    pyautogui.click(999, 333)
    time.sleep(.2)
    load_journal()
    print("journal_3_loaded")
    pyautogui.click(1072, 323)
    time.sleep(.3)
    load_journal()
    print("journal_4_loaded")
    pyautogui.click(1114, 315)
    time.sleep(.3)
    load_journal()
    print("journal_5_loaded")
    pyautogui.click(1142, 338)
    time.sleep(.3)
    load_journal()
    print("journal_6_loaded")
    pyautogui.click(1176, 322)
    time.sleep(.3)
    load_journal()
    print("journal_7_loaded")
    pyautogui.click(1176, 322)
    time.sleep(.3)
    load_journal()
    print("journal_8_loaded")
    pyautogui.click(1163, 323)
    time.sleep(.3)
    load_journal()
    print("journal_9_loaded")
    pyautogui.click(1195, 354)
    time.sleep(.3)
    load_journal()
    print("journal_10_loaded")
    pyautogui.click(1188, 349)
    time.sleep(.3)
    load_journal()
    print("journal_11_loaded")
    pyautogui.click(1196, 335)
    time.sleep(.3)
    load_journal()
    print("journal_12_loaded")
    pyautogui.click(1192, 344)
    time.sleep(.3)
    load_journal()
    print("journal_13_loaded")
    pyautogui.click(1138, 733)
    time.sleep(1.5)
    pyautogui.click(1379, 454)
    time.sleep(.5)
    load_journal()
    print("journal_14_loaded")
    pyautogui.click(1187, 566)
    time.sleep(.3)
    load_journal()
    print("journal_15_loaded")
    pyautogui.click(591, 950)
    time.sleep(2)
    pyautogui.click(340, 808)
    time.sleep(2)
    pyautogui.click(483, 1051)
    time.sleep(2)
    pyautogui.click(264, 988)
    time.sleep(2)
    pyautogui.click(708, 661)
    time.sleep(2)
    pyautogui.click(mouse_targets["bank_choose_tab"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_2_loot_tab"])
    time.sleep(.2)
    pyautogui.click(mouse_targets["bank_move_all_button"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["bank_move_all_button"])
    time.sleep(.2)
    pyautogui.click(mouse_targets["bank_move_all_from_inventory"])
    time.sleep(.5)
    pyautogui.press("esc")
    time.sleep(.2)

    pyautogui.rightClick(675, 435)
    time.sleep(1)
    pyautogui.click(634, 1070)
    time.sleep(2)
    pyautogui.click(480, 539)
    time.sleep(3)

    travel_to_island(configuration.islands["gardener_island_1"])

    pyautogui.click(mouse_targets["enter_traveler_after_join_player_island"])
    time.sleep(.3)
    travel_to_island(configuration.islands["market_guild_fort_sterling"])
    

# debug functions
def check_mouse_click_position():
    def on_press(key):
        try:
            print(f"Key pressed: {key.char} | Keycode: {ord(key.char)}")
        except AttributeError:
            print(f"Special key pressed: {key} | Keycode: {key.value.vk if hasattr(key, 'value') and key.value else 'Unknown'}")

    def on_release(key):
        if key == keyboard.Key.shift_r:  # Stop listener when ESC is pressed
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

def main():
    time.sleep(2)
    #window = window_capture.get_window()
    #win32gui.ShowWindow(window, win32con.SW_RESTORE)
    #win32gui.SetForegroundWindow(window)
    #check_prices_date("caerleon")
    #update_items_price(pandas.read_csv(local_sheet_file_path), "caerleon", False, ["all"])
    #make_buy_orders("lymhurst", ["all"])
    #make_sell_orders()
    #make_market_only_orders()
    #cancel_orders("royal_market")
    #fast_buy_items()
    make_island_cycle()
    #harvest_island("farm")


if __name__ == "__main__":
    main()