class Utility1:

    def probability_cards(self, aces, cards):
        ace_probability = aces / cards
        return ace_probability

    def probability_cardsK(self, king, cards):
        king_probability = king / cards
        return king_probability

    def cards(self):
        cards = 52
        cards_drawn = 1
        cards = cards - cards_drawn
        return cards

    def drawn_aces(self):
        aces_drawn = 1
        aces = 4
        aces = aces - aces_drawn
        return aces

    def count_probHHH(self,len_sample, count_HHH):
        prob_HHH = count_HHH / len_sample
        return prob_HHH

    def length_oneH(self,sample):
        list1 = []
        for temp in sample:
            count = 0
            for char in temp:
                if char == 'H':
                    count += 1
            if count == 1:
                list1.append(temp)
        len_oneH = len(list1)

        print("\n one Head ", len_oneH)

    def prob_oneH(self, len_sample, len_oneH):
        prob_oneH = len_oneH / len_sample
        return prob_oneH

    def at_least_one(self, sample, len_sample,):
        count_TTT = sample.count('TTT')
        print("length of TTT", count_TTT)
        prob_oneHead =(len_sample - count_TTT) / len_sample
        return prob_oneHead

    def at_least_two(self, sample, len_sample):
        list1 = []
        list1 = []
        for temp in sample:
            count = 0
            for char in temp:
                if char == 'H':
                    count += 1
            if count >= 2:
                list1.append(temp)
        len_twoH = len(list1)
        print("\n Two Head ", len_twoH)
        prob_two = len_twoH / len_sample
        return prob_two

    def prob_notRainy_traffic_notlate(self, not_rainy, not_rainy_with_Traffic, not_rainy_with_Traffic_Nolate):
        probability = not_rainy * not_rainy_with_Traffic * not_rainy_with_Traffic_Nolate
        return probability

    
    def probability_cancer(self, true_positive, breast_cancer, false_positive, No_breast_cancer):
        print((true_positive * breast_cancer) / ((true_positive * breast_cancer)
                                                 + (false_positive * No_breast_cancer)))
        prob_cancer = (true_positive * breast_cancer) / (
                    (true_positive * breast_cancer) + (false_positive * No_breast_cancer))
        return prob_cancer
