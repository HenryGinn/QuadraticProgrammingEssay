import matplotlib.pyplot as plt

# Feasible region
x_values = [1, 2, 3, 3, 1]
y_values = [8, 8, 43/6, 9, 9]
plt.fill(x_values, y_values, color="skyblue", alpha=0.5)

# Profit function
radius = pow(68, 0.5)
profit_function = plt.Circle((0, 0), radius=radius, color='red', fill=False)
plt.gca().add_artist(profit_function)

# Plot settings
plt.xlim(1, 3)
plt.ylim(7, 9)
plt.gca().set_aspect('equal')
plt.xticks([1, 1.5, 2, 2.5, 3])
plt.yticks([7, 7.5, 8, 8.5, 9])

# Output
#plt.show()
plt.savefig("Non-global local optimum zoomed.pdf", format="pdf")
