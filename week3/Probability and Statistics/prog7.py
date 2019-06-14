"""7. A bank teller serves customers standing in the queue one by one. Suppose that the service time XiXi for customer
ii has mean EXi=2 (minutes) and Var(Xi)=1. We assume that service times for different bank customers are independent.
Let YY be the total time the bank teller spends serving 50 customers. Write a program to find P(90<Y<110)"""
import math


class Bank:
    
    def __init__(self):
        self.sample = 50
        self. mean = 2
        self.variance = 1
        self.y1 = 90
        self.y2 = 100


    def prob_between(self, Y):
        probability1 = (self.y1 - self.y2)/math.sqrt(self.sample)
        probability2 = (Y - )
