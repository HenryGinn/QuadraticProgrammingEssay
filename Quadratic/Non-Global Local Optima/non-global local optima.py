import matplotlib.pyplot as plt

# Feasible region
x_values = [8, 10, 10, 0, 0, 2, 8]
y_values = [0, 0, 10, 10, 8, 8, 3]
plt.fill(x_values, y_values, color="skyblue", alpha=0.5)

# Profit function
radius = pow(68, 0.5)
profit_function = plt.Circle((0, 0), radius=radius, color='red', fill=False)
plt.gca().add_artist(profit_function)

# Plot settings
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal')

# Output
#plt.show()
plt.savefig("Non-global local optimum.pdf", format="pdf")
