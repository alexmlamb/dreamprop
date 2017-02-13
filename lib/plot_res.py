import matplotlib.pyplot as plt

import sys


def get(fh):
    va = []
    lst = []
    for line in fh:

        if "Valid acc" in line:
            try:
                va.append(float(line.split(" ")[2]))
            except:
                pass

        if "rec" in line:
            try:
                lst.append(float(line.split(" ")[2]))
            except:
                pass

    return va,lst

fh1 = open(sys.argv[1],"r")
fh2 = open(sys.argv[2],"r")

va1,lst1 = get(fh1)
va2,lst2 = get(fh2)


plt.plot(lst1)
plt.plot(lst2)
plt.legend([sys.argv[1], sys.argv[2]],loc="upper right")
plt.title("Reconstruction synthmem penalty")
plt.xlabel('iterations')
plt.show()

plt.plot(va1)
plt.plot(va2)
plt.legend([sys.argv[1], sys.argv[2]],loc="lower right")
plt.ylim([0.90,1.0])
plt.title("Validation Accuracy, " + sys.argv[1] + " = " + str(max(va1)) + ", " + sys.argv[2] + " = " + str(max(va2)))
plt.xlabel('iterations')
plt.show()


