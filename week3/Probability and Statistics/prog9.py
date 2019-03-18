"""9. In a particular pain clinic, 10% of patients are prescribed narcotic pain killers.
Overall, five percent of the clinic’s patients are addicted to narcotics (including pain killers and illegal substances).
 Out of all the people prescribed pain pills, 8% are addicts. If a patient is an addict, write a program to find the
  probability that they will be prescribed pain pills?"""


class ProbPainPills:
    # declare Global Variables

    # P(A)-> 10% of patients are prescribed narcotic pain killers
    pain_killers = 0.1
    # P(B)->five percent of the clinic’s patients are addicted to narcotics
    addicted = 0.05
    # P(B|A) Out of all the people prescribed pain pills, 8% are addicts
    addict_getting_pills = 0.08

    def prescribed_pain(self):
        return (self.addict_getting_pills * self.pain_killers) / self.addicted


obj1 = ProbPainPills()
print("\nProbability that they will be prescribed pain pills: ", obj1.prescribed_pain())
