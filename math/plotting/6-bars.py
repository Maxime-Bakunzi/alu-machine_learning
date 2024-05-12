#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
persons = ['Farrah', 'Fred', 'Felicia']

plt.bar(persons, fruit[0], color=colors[0], label=fruits[0], width=0.5)
for i in range(1, len(fruit)):
    plt.bar(persons, fruit[i], bottom=np.sum(
        fruit[:i], axis=0), color=colors[i], label=fruits[i], width=0.5)

plt.legend()
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.show()
