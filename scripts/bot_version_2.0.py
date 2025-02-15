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

credentials = ServiceAccountCredentials.from_json_keyfile_name('items-prices-albion-credentials.json', configuration.google_sheet_scope)
client = gspread.authorize(credentials)
spreadsheet = client.open("MarketBotAlbion")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
current_platform = platform.system()
if current_platform == "Windows":
    import win32con
    import win32gui
    import windowcapture_windows as windowcapture
    local_sheet_file_path = os.path.join(Path.cwd().parent, "data\PricesCaerleon.csv")
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

current_game_frame = ""
local_sheet_data_frame = pandas.read_csv(local_sheet_file_path)


def get_text_from_screenshot(crop_screenshot_positions, is_gray_reading=True):
    screenshot_file_path = os.path.join(Path.cwd().parent, "images\cropped_zone_screenshot.png")
    gray_screenshot_file_path = os.path.join(Path.cwd().parent, "images\gray_cropped_zone_screenshot.png")
    screenshot = window_capture.get_screenshot(crop_screenshot_positions[0], crop_screenshot_positions[1], crop_screenshot_positions[2], crop_screenshot_positions[3])
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

def get_current_game_frame():
    game_frame = ""
    if get_text_from_screenshot(screenshot_positions["check_game_frame_login"]).replace(" ", "") == "login":
        game_frame = "login"
    elif get_text_from_screenshot(screenshot_positions["check_game_frame_drops_popup"]).replace(" ", "") == "drops":
        game_frame = "drops_popup"
    elif get_text_from_screenshot(screenshot_positions["check_game_frame_characters"]).replace(" ", "") == "characters":
        game_frame = "characters"
    elif get_text_from_screenshot(screenshot_positions["check_escape_menu"]).replace(" ", "") == "logout":
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

    if get_text_from_screenshot(screenshot_positions["check_inventory_open"]).replace(" ", "") != "inventory":
        pyautogui.press("t")
        time.sleep(.3)
    silver_balance_text_inentory = get_text_from_screenshot(screenshot_positions["check_account_silver_from_inventory"], True)
    silver_balance_number_inventory = ConvertSilverToNumber(silver_balance_text_inentory)
    silver_balance_text = get_text_from_screenshot(screenshot_positions["check_account_silver"], True)
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
    pyautogui.typewrite(configuration.login_data[account_name]["email"])
    time.sleep(.1)
    pyautogui.click(mouse_targets["login_password"])
    pyautogui.typewrite(configuration.login_data[account_name]["password"])
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


def open_market():
    return True

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

        text = get_text_from_screenshot(screenshot_positions[f"sell_price_{city_name}"])
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
        while get_text_from_screenshot(screenshot_positions["check_order_exist"]) != "":
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
        while get_text_from_screenshot(screenshot_positions["check_order_exist"]) != "":
            if time.time() - start_time >= 5:
                start_time = time.time()
                pyautogui.scroll(10)
            pyautogui.click(mouse_targets["cancel_order_black_market"])
            time.sleep(.1)

        print("Successfully removed all orders")

def check_statistics_open():
    if get_text_from_screenshot(screenshot_positions["check_buy_orders_title"]) == "buy orders":
        return True
    else:
        pyautogui.click(mouse_targets["extend_item_statistic"])
        time.sleep(.2)

def make_buy_orders(city_name):
    silver_balance = get_account_silver_balance()
    pyautogui.click(mouse_targets["market_create_buy_order_tab"])
    time.sleep(.1)

    worksheet = spreadsheet.get_worksheet(configuration.database_sheets[f"items_to_buy_{city_name}"])
    
    items_to_buy_data = worksheet.get_all_values()
    items_to_buy_data_frame = pandas.DataFrame(items_to_buy_data[1:], columns=items_to_buy_data[0])
    prices_caerleon_data_frame = pandas.read_csv(local_sheet_file_path)
    
    i = 0
    for column_index in range(0, len(items_to_buy_data_frame.columns), 1):
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

                    text = get_text_from_screenshot(screenshot_positions["buy_order_price"])
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

    text = get_text_from_screenshot(screenshot_positions["check_item_price_royal_city"])
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

        avaliable_amount = get_text_from_screenshot(screenshot_positions["check_avaliable_amount"], True)
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

    text = get_text_from_screenshot(screenshot_positions["buy_order_price"], False)
    if text != "":
        if "one" in text:
            item_order_price = 0
        else:
            item_order_price = int("".join(filter(str.isdigit, text)))
    
    text = get_text_from_screenshot(screenshot_positions["sell_order_price"], False)
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

    text = get_text_from_screenshot(screenshot_positions["check_items_in_inventory"]) 
    if text == "":
        if get_text_from_screenshot(screenshot_positions["order_no_longer_exist_check"])  == "someone":
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

                    text = get_text_from_screenshot(screenshot_positions["buy_order_price"])
                    if text != "":
                        try:
                            item_order_price = int("".join(filter(str.isdigit, text)))
                        except ValueError:
                            print("Value Error with item price check")
                    text = get_text_from_screenshot(screenshot_positions["sell_order_price"])
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
                    

def check_mouse_click_position():
    from pynput.mouse import Listener, Button
    def on_click(x, y, button, pressed):
        if pressed and button == Button.left:
            print(f'x={x} and y={y}')
        if pressed and button == Button.right:
            return False
    with Listener(on_click=on_click) as listener:
        listener.join()

def main():
    window = window_capture.get_window()
    win32gui.ShowWindow(window, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(window)
    #check_prices_date("caerleon")
    #update_items_price(pandas.read_csv(local_sheet_file_path), "caerleon", False, ["hoods", "jackets", "shoes", "helmets", "armors", "boots", "shapeshifters"])
    make_buy_orders("lymhurst")
    #make_sell_orders()
    #make_market_only_orders()
    #cancel_orders("black_market")
    #check_mouse_click_position()
    #fast_buy_items()

if __name__ == "__main__":
    main()