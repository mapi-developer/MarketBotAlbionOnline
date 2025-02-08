numbers_configuration = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
window_title = "Albion Online Client"
google_sheet_scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

prices_update_time_gap = 6*60*60
minimum_order_profit_rate = 1.7
minimum_fast_buy_profit_rate = 1.5
blackmarket_fast_sell_price_gap = 0.03
minimum_account_silver_balance = 500000

database_sheets = {
    "caerleon": 0,
    "items_to_buy_lymhurst": 1
}

mouse_targets_1920x1080 = {
    "market_search": [554, 236],
    "market_category": [782, 238],
    "market_tier": [1001, 238],
    "market_tier_1": [1001, 300],
    "market_tier_2": [1001, 333],
    "market_tier_3": [1001, 360],
    "market_tier_4": [1001, 387],
    "market_tier_5": [1001, 418],
    "market_tier_6": [1001, 451],
    "market_tier_7": [1001, 482],
    "market_tier_8": [1001, 516],
    "market_enchantment": [1107, 235],
    "market_enchantment_0": [1107, 300],
    "market_enchantment_1": [1107, 333],
    "market_enchantment_2": [1107, 360],
    "market_enchantment_3": [1107, 387],
    "market_enchantment_4": [1107, 418],
    "market_quality": [1265, 235],
    "market_quality_normal": [1265, 300],
    "market_quality_good": [1265, 333],
    "market_quality_outstanding": [1265, 360],
    "market_search_reset": [689, 232],
    "market_buy_tab": [1451, 316],
    "market_sell_tab": [1443, 406],
    "market_create_buy_order_tab": [1444,482],
    "my_orders_tab": [1444, 574],
    "completed_transactions_tab": [1444, 733],
    "take_all_button": [1313, 922],
    "buy_order_button": [1313, 410],
    "expand_statistic": [1292, 302],
    "change_item_amount_in_order": [529, 582],
    "change_order_price": [589, 646],
    "create_order_button": [873, 755],
    "crate_order_confirmation": [839, 583],
    "close_order_tab": [934, 284],
    "one_silver_less": [518, 647],
    "one_silver_more": [853, 647],
    "sell_fast": [562, 453],
    "sell_long": [562, 483],
    "login_email": [895, 441],
    "login_password": [895, 505],
    "login_button": [1100, 700],
    "character_1": [185, 268],
    "character_2": [185, 507],
    "character_3": [185, 625],
    "enter_world_button": [1200, 883],
    "logout": [1788, 164],
    "lymhurst market": [1159, 349],
    "caerleon": [783, 381],
    "edit_order": [1273, 392],
    "cancel_order": [1352, 385],
    "price_sort_button": [1046, 334],
    "price_highest_button": [1027, 396],
    "price_lowest_button": [1027, 374],
    "extend_item_statistic": [1294, 297],
    "order_no_longer_exist_ok": [957, 540]
}
screenshots_positions_1920x1080 = {
    "buy_order_price": [1296, 340, 1395, 368],
    "sell_order_price": [1025, 338, 1111, 366],
    "sell_price_caerleon": [1023, 407, 1129, 429],
    "check_items_in_inventory": [970, 395, 1010, 425],
    "check_game_frame_login": [922, 385, 1001, 415],
    "check_game_frame_characters": [137, 156, 313, 187],
    "check_game_frame_drops_popup": [1298, 246, 1415, 279],
    "check_escape_menu": [1691, 155, 1764, 178],
    "check_location": [1565, 1027, 1755, 1058],
    "check_account_silver": [1287, 158, 1396, 194],
    "check_order_exist": [650, 370, 813, 421],
    "check_item_price_royal_city": [1080, 405, 1173, 432],
    "check_avaliable_amount": [1110, 346, 1180, 368],
    "activities": [708, 179, 847, 214],
    "check_inventory_open": [1592, 92, 1733, 130],
    "check_account_silver_from_inventory": [1582, 463, 1642, 489],
    "check_buy_orders_title": [1280, 281, 1395, 307],
    "order_no_longer_exist_check": [1081, 440, 1168, 467]
}

mouse_targets_1440x900 = {
    "market_search": [554, 236],
    "market_category": [782, 238],
    "market_tier": [1001, 238],
    "market_tier_1": [1001, 300],
    "market_tier_2": [1001, 333],
    "market_tier_3": [1001, 360],
    "market_tier_4": [1001, 387],
    "market_tier_5": [1001, 418],
    "market_tier_6": [1001, 451],
    "market_tier_7": [1001, 482],
    "market_tier_8": [1001, 516],
    "market_enchantment": [1107, 235],
    "market_enchantment_0": [1107, 300],
    "market_enchantment_1": [1107, 333],
    "market_enchantment_2": [1107, 360],
    "market_enchantment_3": [1107, 387],
    "market_enchantment_4": [1107, 418],
    "market_quality": [1265, 235],
    "market_quality_normal": [1265, 300],
    "market_quality_good": [1265, 333],
    "market_quality_outstanding": [1265, 360],
    "market_search_reset": [689, 232],
    "market_buy_tab": [1451, 316],
    "market_sell_tab": [1443, 406],
    "market_create_buy_order_tab": [1444,482],
    "my_orders_tab": [1444, 574],
    "completed_transactions_tab": [1444, 733],
    "take_all_button": [1313, 922],
    "buy_order_button": [1313, 410],
    "expand_statistic": [1292, 302],
    "change_item_amount_in_order": [529, 582],
    "change_order_price": [589, 646],
    "create_order_button": [873, 755],
    "crate_order_confirmation": [839, 583],
    "close_order_tab": [934, 284],
    "one_silver_less": [518, 647],
    "one_silver_more": [853, 647],
    "sell_fast": [562, 453],
    "sell_long": [562, 483],
    "login_email": [895, 441],
    "login_password": [895, 505],
    "login_button": [1100, 700],
    "character_1": [185, 268],
    "character_2": [185, 507],
    "character_3": [185, 625],
    "enter_world_button": [1200, 883],
    "logout": [1788, 164],
    "lymhurst market": [1159, 349],
    "caerleon": [783, 381],
    "edit_order": [1273, 392],
    "cancel_order": [1352, 385],
    "price_sort_button": [1046, 334],
    "price_highest_button": [1027, 396],
    "price_lowest_button": [1027, 374],
    "extend_item_statistic": [1294, 297],
    "order_no_longer_exist_ok": [957, 540]
}
screenshots_positions_1440x900 = {
    "buy_order_price": [100, 50, 500, 200],
    "sell_order_price": [1025, 338, 1111, 366],
    "sell_price_caerleon": [1023, 407, 1129, 429],
    "check_items_in_inventory": [970, 395, 1010, 425],
    "check_game_frame_login": [922, 385, 1001, 415],
    "check_game_frame_characters": [137, 156, 313, 187],
    "check_game_frame_drops_popup": [1298, 246, 1415, 279],
    "check_escape_menu": [1691, 155, 1764, 178],
    "check_location": [1565, 1027, 1755, 1058],
    "check_account_silver": [1287, 158, 1396, 194],
    "check_order_exist": [650, 370, 813, 421],
    "check_item_price_royal_city": [1080, 405, 1173, 432],
    "check_avaliable_amount": [1110, 346, 1180, 368],
    "activities": [708, 179, 847, 214],
    "check_inventory_open": [1592, 92, 1733, 130],
    "check_account_silver_from_inventory": [1582, 463, 1642, 489],
    "check_buy_orders_title": [1280, 281, 1395, 307],
    "order_no_longer_exist_check": [1081, 440, 1168, 467]
}

qualities_list = {
    "normal",
    "good",
    "outstanding"
}

def get_items_amount(item_price):
    if item_price < 60000:
        items_amount = 2
        if item_price < 40000:
            items_amount = 3
            if item_price <= 20000:
                items_amount = 4
                if item_price <= 10000:
                    items_amount = 8
                    if item_price <= 1000:
                        items_amount = 10
    else:
        items_amount = 1

    return items_amount
