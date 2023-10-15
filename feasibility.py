import numpy as np
from numpy import random


def mat_open(location):
    matrix = np.loadtxt(location)
    matrix = matrix.astype(int)
    (row, coloumn) = (matrix.shape)
    return (matrix, row, coloumn)
cost=np.asarray([5 ,4 ,4 ,3 ,5 ,5 ,3 ,4, 2, 3])

(matrix, row, coloumn)=mat_open(r"C:\Users\Boroush\Documents\University courses\Term 6\Bio-computing\project\geek.txt")

print(matrix)
costs =np.array( [2, 4, 6, 1, 3, 5])

# Binary matrix showing column coverage
covers = matrix

# Initialize vectors
solution = [[0, np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])], [0, np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])], [0, np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])]]

# ai generation

def feasability(matrix,soloutions,row,):
    row_coverage = []
    for r in range(row):
        b= matrix[r,]
        c=list( np.where(b == 1)[0])
        row_coverage.append(c)
#print("row",row_coverage)

    wi=[]
    w=0


    tw=[]
    for chr in solution:
        chr=chr[1]
        columns_of_chr=np.where(chr == 1)[0]
        for row in row_coverage:
            row_c_count=0
            for c in coloumns_of_chr:
                row_c_count += row.count(c)
            wi.append(row_c_count)
        tw.append(wi)
        wi=[]


print("tw",tw)




#for i in range(row):
