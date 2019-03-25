import numpy as np


def validate_num(num):
    try:
        int(num)
        return True
    except Exception:
        return False


# creating list for all types and len > 2
def create_list_all(element):
    li = []
    try:
        for e in range(element):
            input2 = input("Enter element : ")
            if len(input2) >= 2:
                li.append(input2)
            else:
                print("String should be len 2 or more")
    except Exception as e:
        print(e)
    return li


# get random distribution of x and y
def get_random_distribution():
    value = 0
    x = input("Enter random distribution in x:")
    # validate random value for x
    validate = validate_num(x)
    if validate:
        x = int(x)
        # generate array of random number
        value = np.random.randn(x)
    return value


# get  marks
def get_marks(length):
    mark_list = []
    try:
        for e in range(length):
            input1 = int(input("Enter marks : "))
            valid = validate_num(input1)
            if valid:
                mark_list.append(input1)
            else:
                print("enter number only")
    except Exception as e:
        print(e)
    return mark_list
