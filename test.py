# extract all lines as a list
lines = [line.rstrip('\n') for line in open('test.txt', 'r')]

# set two variable counters
counter_6 = 0
counter_14 = 0

for line in lines:
    # split the lines into the needed parts
    password_part = line.split(", ")[1]

    # increment for Password < 6
    if "< 6" in password_part:
        counter_6 += 1
    # increment for Password > 14
    elif "> 14" in password_part:
        counter_14 += 1

print(counter_6, counter_14)
