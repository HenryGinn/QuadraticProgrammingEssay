from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
fig = plt.figure()
ax = plt.axes(projection='3d',computed_zorder=False)
fig.add_axes(ax)
ax.view_init(elev=14, azim=11, roll=0)

# Fixing axis ratio and limits
limit = 1
ax.set_box_aspect([1,1,1])
ax.set_xlim3d([0, limit])
ax.set_ylim3d([0, limit])
ax.set_zlim3d([0, limit])

# Removing axis details
blank = (1, 1, 1, 0)
ax.xaxis.line.set_color(blank)
ax.yaxis.line.set_color(blank)
ax.zaxis.line.set_color(blank)
ax.tick_params(color=blank, labelcolor=blank)

# Plane
X, Y, Z = 1, 1, 1
a, b, c = 1, 1, 1
p = 1.732
x1 = (0, (p-c*Z)/b, c*Z)
x2 = (0, b*Y, (p-b*Y)/c)
y1 = (a*X, 0, (p-a*X)/c)
y2 = ((p-c*Z)/a, 0, c*Z)
z1 = ((p-b*Y)/a, b*Y, 0)
z2 = (a*X, (p-a*X)/b, 0)
vertices = [x2, x1, y2, y1, z2, z1]
poly = Poly3DCollection([vertices], alpha=0.5, facecolors="skyblue", zorder=2)
ax.add_collection3d(poly)

# Create a sphere
r = 1
N = 300
theta, phi = np.mgrid[0:0.5*np.pi:N*1j, 0:0.5*np.pi:N*1j]
x = r*np.sin(theta) * np.cos(phi)
y = r*np.sin(theta) * np.sin(phi)
z = r*np.cos(theta)
dot_product = a*x + b*y + c*z
dot_product_limit = p
below_plane = dot_product < dot_product_limit

# Plotting two parts of the sphere
x_below = np.where(below_plane, x, np.nan)
y_below = np.where(below_plane, y, np.nan)
z_below = np.where(below_plane, z, np.nan)
ax.plot_surface(x_below, y_below, z_below, color="red", alpha=0.5, zorder=1)

x_above = np.where(below_plane, np.nan, x)
y_above = np.where(below_plane, np.nan, y)
z_above = np.where(below_plane, np.nan, z)
ax.plot_surface(x_above, y_above, z_above, color="red", alpha=0.5, zorder=3)

# Intersection between plane and sphere
radius = np.sqrt(r**2 - p**2/(a**2 + b**2 + c**2))
v0 = p/3 * np.array([a, b, c])
v1 = np.array([b*c, a*c, -2*a*b])
v2 = np.array([-2*a*b**2 - a*c**2,
               b*c**2 + 2*a**2*b,
               a**2*c - b**2*c])
v1 = radius * v1 / np.linalg.norm(v1)
v2 = radius * v2 / np.linalg.norm(v2)
angle = np.linspace(0, 2*np.pi, 100)
circle = v0 + np.outer(np.sin(angle), v1) + np.outer(np.cos(angle), v2)
x, y, z = np.transpose(circle)
plt.plot(x, y, z, color="black", linewidth=2, zorder=4)

# Output
plt.show()
#plt.savefig("Intersection Circle of Plane and Sphere Touching.pdf", format="pdf", bbox_inches='tight', pad_inches=-0.3)
#plt.savefig("Intersection Circle of Plane and Sphere Small Circle.pdf", format="pdf", bbox_inches='tight', pad_inches=-0.3)
#plt.savefig("Intersection Circle of Plane and Sphere Large Circle.pdf", format="pdf", bbox_inches='tight', pad_inches=-0.3)
