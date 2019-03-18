"""11. A radar unit is used to measure speeds of cars on a motorway.
The speeds are normally distributed with a mean of 90 km/hr and a standard deviation of 10 km/hr.
Write a program to find the probability that a car picked at random is travelling at more than 100 km/hr? """


class Radar:
    def __init__(self):
        self.mean = 90
        self.sd = 10
        self.sample = 100
        self.area = 1
        self.z_area = 0.8413

    def radar_prob(self):
        z_val = (self.sample - self.mean) / self.sd
        print("\n Z Value", z_val)
        total_area = self.area - self.z_area
        print("\n total area :", total_area)


object1 = Radar()
# object1.radar_prob()

print("\n1.  probability that a car picked at random is travelling at more than 100 km/hr ")
print("_______________________________________________________________________________________________________________")
while 1:
    try:

        ch = input("\n Enter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:
                object1.radar_prob()

            elif choice == 2:
                exit()
        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")
