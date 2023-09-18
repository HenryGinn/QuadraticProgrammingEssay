import matplotlib.pyplot as plt

# Line settings
colour = "blue"
linewidth = 1

# Lines
plt.plot([2.6, 3.6], [0, 5], color=colour, linewidth=linewidth)
plt.plot([3.5, 2.25], [0, 5], color=colour, linewidth=linewidth)
plt.plot([4.5, 0.75], [0, 5], color=colour, linewidth=linewidth)
plt.plot([0, 5], [4.25, 0.5], color=colour, linewidth=linewidth)
plt.plot([0, 5], [3.2, 1.2], color=colour, linewidth=linewidth)

# Objective Function
plt.plot([0, 2.6], [0.25, 0], color="red", linewidth=linewidth)

# Infeasible region
alpha = 0.5
x_values = [2.6, 5, 5, 0, 0, 3]
y_values = [0, 0, 5, 5, 3.2, 2]
plt.fill(x_values, y_values, color="skyblue", alpha=alpha)

# Line labels
fontsize = 16
plt.text(0.3, 3.2, "5", fontsize=fontsize)
plt.text(0.3, 4.2, "4", fontsize=fontsize)
plt.text(1.2, 4.6, "3", fontsize=fontsize)
plt.text(2.5, 4.6, "2", fontsize=fontsize)
plt.text(3.7, 4.6, "1", fontsize=fontsize)

# Plot settings
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.gca().set_aspect('equal')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

# Output
#plt.show()
plt.savefig("Ratio is equal.pdf", format="pdf")
