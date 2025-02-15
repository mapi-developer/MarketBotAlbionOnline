from random import random

x = {}
days_number = 10000000000

def check_day_rates(day_number):
    def get_random_chances():
        if random() < 0.93:
            return "seed"
        else:
            return "nothing"
        
    seeds = 0
    nothing = 0

    for i in range(0, 144, 1):
        lol = get_random_chances()
        if lol == "seed":
            seeds = seeds + 1
        else:
            nothing = nothing + 1

    if seeds <= 101:
        print(f"всего 100 семечек на {day_number} день")

    return seeds, nothing

total_seeds = 0
total_nothing = 0

for i in range(0, days_number, 1):
    seeds, nothing = check_day_rates(i)
    x[f"day_{i}"] = [seeds, nothing]
    total_seeds = total_seeds + seeds
    total_nothing = total_nothing + nothing

total_seeds_rate = total_seeds/days_number
total_nothing_rate = total_nothing/days_number

print(f"{total_seeds_rate}, {total_nothing_rate} --- chances:({total_seeds_rate/144*100} : {total_nothing_rate/144*100})")