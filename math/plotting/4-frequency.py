#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

x = student_grades 

z = np.arange(0, 110, 10)
plt.hist(x ,bins = z, edgecolor= "black")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xticks(z)
plt.title("Project A")
plt.show()
