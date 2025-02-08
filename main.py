import cv2 as cv
import os
import time
from scripts.windowcapture_windows import WindowCapture
import pyautogui
import bot_config
import json
import pytesseract
import copy
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
numbers_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

data_file_path = "bot_data.json"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1920x1080
wincap = WindowCapture("Albion Online Client")
current_game_frame = ""

def ConvertSilverToNumber(string):
    string = string.strip().replace(" ", "")  # Normalize case and remove extra spaces
    multipliers = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}

    match = re.match(r'([\d\.]+)([kmb]?)$', string)  # Extract number and suffix
    if match:
        number, suffix = match.groups()
        return int(float(number) * multipliers.get(suffix, 1))  # Convert and multiply
    else:
        return "can't check silver"

def GetAccountSilverBalance():
    silver_balance_text = GetTextFromScreenshot(bot_config.screenshots_positions["check_account_silver"])
    silver_balance_number = ConvertSilverToNumber(silver_balance_text)
    return silver_balance_number

def GetTextFromScreenshot(screenshot_positions):
    screenshot = wincap.get_screenshot(screenshot_positions[0], screenshot_positions[1], screenshot_positions[2], screenshot_positions[3])
    screenshot.save("screenshot.png")
    screenshot = cv.imread("screenshot.png")
    gray_screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    gray_screenshot = cv.threshold(gray_screenshot, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imwrite("gray_screenshot.png", gray_screenshot)
    text = pytesseract.image_to_string(screenshot, config="--psm 6").rstrip("\n").lower()
    return text

def CancelOrdersOnMarket():
    pyautogui.click(bot_config.mouse_targets["my_orders_tab"])
    time.sleep(.2)
    start_time = time.time()
    while GetTextFromScreenshot(bot_config.screenshots_positions["check_order_exist"]) != "":
        if time.time() - start_time >= 5:
            start_time = time.time()
            pyautogui.scroll(10)
        pyautogui.click(bot_config.mouse_targets["cancel_order"])
        time.sleep(.1)

    print("Successfully removed all orders")

def MakeOrderOnItem(data, item_full_title, item_category):
    item_name, item_tier, item_enchantment = item_full_title.split("_")
    item_amount = 1
    item_order_price = 0
    have_enough_silver = True
    canMakeOrder = False

    pyautogui.click(bot_config.mouse_targets["market_search_reset"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["market_search"])
    pyautogui.typewrite(item_name)
    pyautogui.click(bot_config.mouse_targets["market_tier"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_tier_{item_tier}"])
    pyautogui.click(bot_config.mouse_targets["market_enchantment"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_enchantment_{item_enchantment}"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["buy_order_button"])
    time.sleep(.1)

    text = GetTextFromScreenshot(bot_config.screenshots_positions["buy_order_price"])
    if text != "":
        try:
            item_order_price = int("".join(filter(str.isdigit, text)))
            if item_order_price * bot_config.minimum_profit_rate < data["items_prices_caerleon"][item_category][item_full_title]:
                canMakeOrder = True
        except ValueError:
            print("Value Error with item price check")

    item_amount = bot_config.GetItemsAmount(item_order_price)

    silver_to_buy = item_amount * item_order_price * 1.05
    silver_on_account = GetAccountSilverBalance()
    if silver_on_account == "can't check silver":
        print("Can't check silver")
    elif silver_on_account < silver_to_buy:
        have_enough_silver = False

    if canMakeOrder == True and have_enough_silver == True:
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets["change_item_amount_in_order"])
        pyautogui.typewrite(str(item_amount))
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets["one_silver_more"])
        pyautogui.click(bot_config.mouse_targets["create_order_button"])
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets["crate_order_confirmation"])
        time.sleep(.1)
        print(f"Made order on {item_name} {item_tier}.{item_enchantment}")
        return True
    else:
        if have_enough_silver == False:
            return False
        pyautogui.click(bot_config.mouse_targets["close_order_tab"])
        time.sleep(.1)
        print(f"Can't make an order on {item_name} {item_tier}.{item_enchantment}")
        return True

def CheckItemPrice(item_full_title, city_name):
    item_name, item_tier, item_enchantment = item_full_title.split("_")
    best_item_price = 0

    pyautogui.click(bot_config.mouse_targets["market_search_reset"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["market_search"])
    pyautogui.typewrite(item_name)
    pyautogui.click(bot_config.mouse_targets["market_tier"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_tier_{item_tier}"])
    pyautogui.click(bot_config.mouse_targets["market_enchantment"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_enchantment_{item_enchantment}"])

    for i in bot_config.qualities_list:
        pyautogui.click(bot_config.mouse_targets["market_quality"])
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets[f"market_quality_{i}"])
        time.sleep(.1)

        text = GetTextFromScreenshot(bot_config.screenshots_positions[f"sell_price_{city_name}"])
        if text != "":
            try:
                item_price = int("".join(filter(str.isdigit, text)))
                if item_price > best_item_price:
                    best_item_price = item_price  
            except ValueError:
                print("Value Error with item price check")       

    return best_item_price

def MakeSellOrderOnItem():
    pyautogui.click(bot_config.mouse_targets["buy_order_button"])
    time.sleep(.1)

    item_order_price, sell_item_price = 0, 0

    text = GetTextFromScreenshot(bot_config.screenshots_positions["buy_order_price"])
    if text != "":
        if "ONE" in text:
            item_order_price = 0
        else:
            item_order_price = int("".join(filter(str.isdigit, text)))
    
    text = GetTextFromScreenshot(bot_config.screenshots_positions["sell_order_price"])
    if text != "":
        try:
            sell_item_price = int("".join(filter(str.isdigit, text)))
        except ValueError:
            print("Value Error with item price check")

    price_gap = sell_item_price * bot_config.blackmarket_price_gap
    if sell_item_price - price_gap <= item_order_price:
        pyautogui.click(bot_config.mouse_targets["sell_fast"])
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets["create_order_button"])
    else:
        pyautogui.click(bot_config.mouse_targets["sell_long"])
        time.sleep(.1)
        pyautogui.click(bot_config.mouse_targets["one_silver_less"])
        pyautogui.click(bot_config.mouse_targets["create_order_button"])

    time.sleep(.1)
 
def MakeBuyOrders(city_name):
    have_silver_for_new_order = True
    pyautogui.click(bot_config.mouse_targets["market_create_buy_order_tab"])
    time.sleep(.1)

    with open(data_file_path, "r") as content_data:
        data = json.load(content_data)
    items_to_buy_list = data[f"items_to_buy_list_{city_name}"]

    for item_category in items_to_buy_list:
        for i in items_to_buy_list[item_category]:
            if have_silver_for_new_order == False:
                print("No more silver on balance")
                return
            have_silver_for_new_order = MakeOrderOnItem(data, i, item_category)
    time.sleep(.2)
    print("Orders maded")

def UpdateItemsPrice(city_name, only_zero_price=False, items_categories_to_check=["all"]):
    pyautogui.click(bot_config.mouse_targets["market_buy_tab"])
    time.sleep(.1)

    with open(data_file_path, "r") as bot_data:
        data = json.load(bot_data)
    items_prices_in_city = copy.deepcopy(data[f"items_prices_{city_name}"])
    bot_data.close()

    if items_categories_to_check[0] == "all":
        for item_category in items_prices_in_city:
            for i in items_prices_in_city[item_category]:
                if only_zero_price:
                    if items_prices_in_city[item_category][i] == 0:
                        items_prices_in_city[item_category][i] = CheckItemPrice(i, city_name)
                else:
                    items_prices_in_city[item_category][i] = CheckItemPrice(i, city_name)
    else:
        for item_category in items_categories_to_check:
            for i in items_prices_in_city[item_category]:
                if only_zero_price:
                    if items_prices_in_city[item_category][i] == 0:
                        items_prices_in_city[item_category][i] = CheckItemPrice(i, city_name)
                else:
                    items_prices_in_city[item_category][i] = CheckItemPrice(i, city_name)

    with open(data_file_path, "w") as bot_data:
        data[f"items_prices_{city_name}"] = items_prices_in_city
        json.dump(data, bot_data, indent=4, separators=(",", ": "))
    bot_data.close()
    time.sleep(.2)
    print("Prices updated successfully") 

def MakeSellOrders():
    pyautogui.click(bot_config.mouse_targets["market_sell_tab"])
    time.sleep(.1)

    text = GetTextFromScreenshot(bot_config.screenshots_positions["check_items_in_inventory"]) 
    if text == "":
        print("Selling orders done successfully")
        return True
    else:
        MakeSellOrderOnItem()
        return MakeSellOrders()

def CheckMouseClick():
    from pynput.mouse import Listener, Button

    # Function called on a mouse click
    def on_click(x, y, button, pressed):
        # Check if the left button was pressed
        if pressed and button == Button.left:
            # Print the click coordinates
            print(f'x={x} and y={y}')


    # Initialize the Listener to monitor mouse clicks
    with Listener(on_click=on_click) as listener:
        listener.join()

def Logout():
    global current_game_frame
    current_game_frame = GetCurrentGameFrame()
    if current_game_frame == "login":
        return
    while current_game_frame != "escape_menu":
        pyautogui.press("esc")
        current_game_frame = GetCurrentGameFrame()
        time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["logout"])
    time.sleep(12)
    while current_game_frame != "login":
        current_game_frame = GetCurrentGameFrame()
        time.sleep(.3)
    time.sleep(.1)

def GetCurrentGameFrame():
    game_frame = ""
    if GetTextFromScreenshot(bot_config.screenshots_positions["check_game_frame_login"]).replace(" ", "") == "login":
        print(GetTextFromScreenshot(bot_config.screenshots_positions["check_game_frame_login"]).replace(" ", ""))
        game_frame = "login"
    elif GetTextFromScreenshot(bot_config.screenshots_positions["check_game_frame_drops_popup"]).replace(" ", "") == "drops":
        game_frame = "drops_popup"
    elif GetTextFromScreenshot(bot_config.screenshots_positions["check_game_frame_characters"]).replace(" ", "") == "characters":
        game_frame = "characters"
    elif GetTextFromScreenshot(bot_config.screenshots_positions["check_escape_menu"]).replace(" ", "") == "logout":
        game_frame = "escape_menu"

    return game_frame

def GetCurrentLocation():
    location = re.sub(r'[^a-zA-Z\s]', '', GetTextFromScreenshot(bot_config.screenshots_positions["check_location"]))
    return re.sub(r'\s$', '', location)

def ChangeAccount(account_name, characer_number=1):
    global current_game_frame
    current_game_frame = GetCurrentGameFrame()
    if current_game_frame != "login":
        Logout()
    pyautogui.click(bot_config.mouse_targets["login_email"])
    pyautogui.typewrite(bot_config.login_data[account_name]["email"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["login_password"])
    pyautogui.typewrite(bot_config.login_data[account_name]["password"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["login_button"])
    time.sleep(2)
    while current_game_frame != "characters":
       current_game_frame = GetCurrentGameFrame()
       time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"character_{characer_number}"])
    pyautogui.click(bot_config.mouse_targets["enter_world_button"])
    time.sleep(5)
    print(f"Successfully logged into {account_name}")

def OpenMarket(market_name):
    while GetCurrentLocation() != market_name:
        time.sleep(.5)
    pyautogui.click(bot_config.mouse_targets[market_name])
    time.sleep(.2)
    print(f"Successfully open {market_name}")

def BuyItemFromMarket(data, item_full_title, item_category):
    item_name, item_tier, item_enchantment = item_full_title.split("_")
    item_price = 0
    have_enough_silver = True
    canMakeOrder = False
    avaliable_amount = 1

    pyautogui.click(bot_config.mouse_targets["market_search_reset"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets["market_search"])
    pyautogui.typewrite(item_name)
    pyautogui.click(bot_config.mouse_targets["market_tier"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_tier_{item_tier}"])
    pyautogui.click(bot_config.mouse_targets["market_enchantment"])
    time.sleep(.1)
    pyautogui.click(bot_config.mouse_targets[f"market_enchantment_{item_enchantment}"])
    time.sleep(.6)

    text = GetTextFromScreenshot(bot_config.screenshots_positions["check_item_price_royal_city"])
    if text != "":
        try:
            item_price = int("".join(filter(str.isdigit, text)))
            if item_price * bot_config.minimum_profit_rate_fast < data["items_prices_caerleon"][item_category][item_full_title]:
                canMakeOrder = True
        except ValueError:
            print("Value Error with item price check")

    if canMakeOrder == True:
        pyautogui.click(bot_config.mouse_targets["buy_order_button"])
        time.sleep(.2)

        avaliable_amount = GetTextFromScreenshot(bot_config.screenshots_positions["check_avaliable_amount"])
        try:
            avaliable_amount = int("".join(filter(str.isdigit, avaliable_amount)))
        except ValueError:
            avaliable_amount = 1

        silver_to_buy = item_price * avaliable_amount * 1.05
        silver_on_account = GetAccountSilverBalance()
        if silver_on_account == "can't check silver":
            print("Can't check silver")
        elif silver_on_account < silver_to_buy:
            have_enough_silver = False

    if canMakeOrder == True and have_enough_silver == True:
        time.sleep(.1)
        if avaliable_amount > 1:
            pyautogui.click(bot_config.mouse_targets["change_item_amount_in_order"])
            pyautogui.typewrite(str(avaliable_amount))
        pyautogui.click(bot_config.mouse_targets["create_order_button"])
        time.sleep(.1)
        print(f"Successfully bought {item_name} {item_tier}.{item_enchantment}")
        return BuyItemFromMarket(data, item_full_title, item_category)
    else:
        if have_enough_silver == False:
            return False
        pyautogui.click(bot_config.mouse_targets["close_order_tab"])
        time.sleep(.1)
        print(f"Can't fast buy {item_name} {item_tier}.{item_enchantment}")
        return True

def FastBuyItems(city_name):
    have_silver_for_new_order = True
    pyautogui.click(bot_config.mouse_targets["market_buy_tab"])
    time.sleep(.1)

    with open(data_file_path, "r") as content_data:
        data = json.load(content_data)
    items_to_buy_list = data[f"items_to_buy_list_{city_name}_fast"]

    for item_category in items_to_buy_list:
        for i in items_to_buy_list[item_category]:
            if have_silver_for_new_order == False:
                print("No more silver on balance")
                return
            have_silver_for_new_order = BuyItemFromMarket(data, i, item_category)

    time.sleep(.2)
    print("Have done fast buying items")

def TestFunction():
    time.sleep(2)
    #FastBuyItems("lymhurst")````
    #CheckMouseClick()
    #ChangeAccount("Matvey4aAlt1")
    #OpenMarket("lymhurst market")
    #UpdateItemsPrice("caerleon", False, ["all"])
    #print(GetAccountSilverBalance())
    #CancelOrdersOnMarket()
    #MakeBuyOrders("lymhurst")
    #MakeSellOrders()

    


#CheckMouseClick()
#MakeBuyOrders("lymhurst")
#UpdateItemsPrice("caerleon", True, ["jackets", "hoods", "sandals", "robes", "cowls"])
#MakeSellOrders()
#start()
#print(GetCurrentGameFrame())
#print(GetCurrentCity())
#ChangeAccount("Matvey4aAlt1")
#OpenMarket("lymhurst market")

TestFunction()
