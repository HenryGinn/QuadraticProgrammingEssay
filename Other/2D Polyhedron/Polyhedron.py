import matplotlib.pyplot as plt

alpha = 0.5
colour = "skyblue"

# Bottom left
x_values = [0, 5, 0]
y_values = [0, 0, 4]
plt.fill(x_values, y_values, colour, alpha=alpha)

# Bottom right
x_values = [6, 2, 6]
y_values = [0, 0, 3]
plt.fill(x_values, y_values, colour, alpha=alpha)

# Top left
x_values = [0, 0, 5]
y_values = [5, 3, 5]
plt.fill(x_values, y_values, colour, alpha=alpha)

# Top right
x_values = [6, 4, 4.5, 6]
y_values = [5, 5, 0, 0]
plt.fill(x_values, y_values, colour, alpha=alpha)

plt.xlim(0, 6)
plt.ylim(0, 5)

plt.savefig("Polyhedron.pdf", format="pdf")
