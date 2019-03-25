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
            if len(input2) >= 1:
                li.append(input2)
            else:
                print("String should be len 2 or more")
    except Exception as e:
        print(e)
    return li

