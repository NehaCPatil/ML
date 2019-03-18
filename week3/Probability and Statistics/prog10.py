"""10.  X is a normally normally distributed variable with mean μ = 30 and standard deviation σ = 4. Write a program to find
a. P(x < 40)
b. P(x > 21)
c. P(30 < x < 35)"""


class Normal:
    def __init__(self):
        self.mean = 30
        self.sd = 4
        self.z_1 = 0.9938
        self.z_2 = 0.0122
        self.z_3 = 0.8944

    """a. P(x < 40)"""

    def z_value_x(self):
        x_value = 40
        z_val = (x_value - self.mean) / self.sd
        print("\n Value of Z: ", z_val)
        if z_val <= 2.5:
            print("\n Area to the left side:", self.z_1)

    """b. P(x > 21)"""

    def z_value_x1(self):
        x1_value = 21
        z_val = (x1_value - self.mean) / self.sd
        print("\n Value of Z:", z_val)
        if z_val <= -2.25:
            area = 1
            print("\n Area to the left side", self.z_2)
            total_area = area - self.z_2
            print("\n total area :", total_area)

    """c. P(30 < x < 35)"""

    def z_value_x2(self):
        x2_value = 30
        x3_value = 35
        z_val = (x2_value - self.mean) / self.sd
        print("\n Value of Z:", z_val)

        z_val1 = (x3_value - self.mean) / self.sd
        print("\n Value of Z:", z_val1)
        if z_val < 1.25:
            area_half = 0.5
            print("\n Area to the left side:", self.z_3)
            total_area = self.z_3 - area_half
            print("\n total area:", total_area)


object1 = Normal()

print("\n X is a normally normally distributed variable with mean μ = 30 and standard deviation σ = 4")
print("\n1.  P(x < 40)  \n ""2.  P(x > 21)  \n ""3.  P(30 < x < 35) \n""4. EXIT ")
while 1:
    try:

        
        ch = input("\n Enter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:
                object1.z_value_x()

            elif choice == 2:
                object1.z_value_x1()

            elif choice == 3:

                object1.z_value_x2()

            elif choice == 4:

                exit()
        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")

