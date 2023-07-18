import matplotlib.pyplot as plt

x_values = [-5, -5, 10]
y_values = [-4, 8, -4]
colour = "skyblue"
plt.fill(x_values, y_values, colour)

plt.xlim(0, 6)
plt.ylim(0, 5)

plt.savefig("HalfSpace.pdf", format="pdf")
