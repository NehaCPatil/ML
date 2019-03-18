class Utility1:

    def __init__(self):
        self.mat_list1 = [[12, 7, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

        self.mat_list2 = [[5, 8, 1],
                          [6, 7, 3],
                          [4, 5, 9]]

        # display matrix

    def read_matrix1(self):
        print("mat_list:")
        for temp in self.mat_list1:
            print(temp)

    def read_matrix2(self):
        print("\nmat_list1:")
        for temp1 in self.mat_list2:
            print(temp1)

        # method for addition

    def add_matrix(self, mat_list1, mat_list2):
        while 1:
            try:
                print("\n 1. Get Matrix add""\n""2. Exit")
                ch = input("Enter choice")

                choice = int(ch)

                if ch.isdigit():

                    if choice == 1:

                        result_list = [[mat_list1[row][col] + mat_list2[row][col] for col in range(len(mat_list1[0]))]
                                       for row in range(len(mat_list1))]
                        print("\n Addition Matrix:-")
                        for temp1 in result_list:
                            print(temp1)

                    elif choice == 2:
                        exit()
                else:
                    raise Exception

            except Exception:
                print("\n Invalid Input")

    def scalar_Product_Mat(self, mat_list1, scalar):
        while 1:
            try:
                print("\n 1. Get Matrix ""\n""2. Exit")
                ch = input("Enter choice")

                choice = int(ch)

                if ch.isdigit():

                    if choice == 1:
                        print("\n scalar multiplication of matrix and a number:")
                        for row in range(len(mat_list1)):
                            for col in range(len(mat_list1[0])):
                                mat_list1[row][col] = mat_list1[row][col] * scalar

                        for temp1 in mat_list1:
                            print(temp1)
                    elif choice == 2:
                        exit()
                else:
                    raise Exception

            except Exception:

                print("\n Invalid Input")

    def vector_multiply_mat(self, matrix_X, vector_Y):
        m = len(matrix_X)
        v = len(vector_Y)
        list_new = [0] * m
        sum1 = 0
        for temp2 in range(m):
            row1 = matrix_X[temp2]
            for temp1 in range(v):
                sum1 += sum(row1[temp1] * vector_Y[temp1])
            list_new[temp2] = sum1
            sum1 = 0
        return list_new

    def mul_matrix(self, mat_list1, mat_list2, result_list):
        for i in range(len(mat_list1)):
            for j in range(len(mat_list1[0])):
                result_list[i][j] += mat_list1[i][j] * mat_list2[i][j]
        print("\n Multiplication matrix")
        for temp1 in result_list:
            print(temp1)

    def transpose_mat(self, mat_list1):
        while 1:
            try:
                print("\n 1. Get Matrix ""\n""2. Exit")
                ch = input("Enter choice")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        result = [[mat_list1[j][i] for j in range(len(mat_list1))] for i in
                                  range(len(mat_list1[0]))]
                        print("\n\n transpose matrix of matrix Y:")
                        for temp1 in result:
                            print(temp1)
                    elif choice == 2:
                        exit()
                else:
                    raise Exception

            except Exception:
                print("Invalid choice")
