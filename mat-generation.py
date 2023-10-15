#x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))

import numpy as np
from numpy import random
def mat_generation(row,column,density,name):
    mat = random.choice([1, 0], p=[density,(1-density)], size=(row,column))
    name=name+".txt"
    h=str(row)+"*"+str(column)+"  "+str(density)
    save_2_file = np.savetxt(name, mat, fmt='%i', header=h)
    return
mat_generation(200,1000,0.02,"4")
mat_generation(200,2000,0.02,"5")
mat_generation(200,1000,0.05,"6")
mat_generation(300,3000,0.02,"A")
mat_generation(300,3000,0.05,"B")
mat_generation(400,4000,0.02,"C")
mat_generation(400,4000,0.05,"D")
mat_generation(500,5000,0.1,"E")
mat_generation(500,5000,0.2,"F")
mat_generation(1000,1000,0.02,"G")
mat_generation(1000,1000,0.05,"H")