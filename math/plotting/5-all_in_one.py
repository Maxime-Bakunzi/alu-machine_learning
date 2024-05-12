#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data for plots
y0 = np.arange(0, 11) ** 3
x0 = np.arange(0, 11)
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot 1
axs[0, 0].plot(x0, y0, color="red")
axs[0, 0].set_title("Plot 1", fontsize='x-small')
axs[0, 0].set_xlabel('X-axis', fontsize='x-small')
axs[0, 0].set_ylabel('Y-axis', fontsize='x-small')


# Plot 2
axs[0, 1].scatter(x1, y1, s=10, c="purple")
axs[0, 1].set_title("Plot 2", fontsize='x-small')
axs[0, 1].set_xlabel('Height (in)', fontsize='x-small')
axs[0, 1].set_ylabel('Weight (lbs)', fontsize='x-small')

# Plot 3
axs[1, 0].plot(x2, y2)
axs[1, 0].set_title("Plot 3", fontsize='x-small')
axs[1, 0].set_xlabel("Time (years)", fontsize='x-small')
axs[1, 0].set_ylabel("Fraction Remaining", fontsize='x-small')
axs[1, 0].set_yscale('log')

# Plot 4
axs[1, 1].plot(x3, y31, color="red", linestyle="--", label="C-14")
axs[1, 1].plot(x3, y32, color="green", label="Ra-226")
axs[1, 1].set_title("Plot 4", fontsize='x-small')
axs[1, 1].set_xlabel("Time (years)", fontsize='x-small')
axs[1, 1].set_ylabel("Fraction Remaining", fontsize='x-small')
axs[1, 1].legend()

# Plot 5
z = np.arange(0, 110, 10)
axs[2, 0].hist(student_grades, bins=z, edgecolor="black")
axs[2, 0].set_title("Plot 5", fontsize='x-small')
axs[2, 0].set_xlabel("Grades", fontsize='x-small')
axs[2, 0].set_ylabel("Number of Students", fontsize='x-small')
axs[2, 0].set_xlim(0, 100)
axs[2, 0].set_ylim(0, 30)
axs[2, 0].set_xticks(z)

# Removing empty subplot
fig.delaxes(axs[2, 1])

# Adjust layout
plt.tight_layout()

# Overall title
fig.suptitle("All in One", fontsize='x-small')

# Show plot
plt.show()
