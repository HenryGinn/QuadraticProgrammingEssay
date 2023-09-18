import matplotlib.pyplot as plt

# Line settings
colour = "skyblue"
alpha = 0.5

n = 10
a = -1
b = 3

# Lines
plt.fill([0, -n, -n, 0], [a, a, b, b], color=colour, alpha=alpha) # y >= 0
plt.fill([-n, n, n, -n], [0, 0, a, a], color=colour, alpha=alpha) # x >= 0
plt.fill([n - 1, n - 0.5, n, n], [a, b, b, a], color=colour, alpha=alpha) # Other constraint
#plt.fill([-n, n, n, -n], [b, b, b - 1.5, b - 0.5], color=colour, alpha=alpha) # alpha < 0 constraint
#plt.fill([-n, n, n, -n], [b, b, b - 1, b - 1], color=colour, alpha=alpha) # alpha = 0 constraint
plt.fill([-n, n, n, -n], [b, b, b - 0.5, b - 1.5], color=colour, alpha=alpha) # alpha > 0 constraint

# Plot settings
plt.xlim(-n, n)
plt.ylim(a, b)
plt.gca().set_aspect('equal')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

# Output
#plt.show()
#plt.savefig("Alpha is 0 (less than).pdf", format="pdf")
#plt.savefig("Alpha is 0 (equal).pdf", format="pdf")
plt.savefig("Alpha is 0 (greater than).pdf", format="pdf")
