# %%
import matplotlib.pyplot as plt
import numpy as np
from bez.bezier import interpolate_bezier, fit_bezier, std_distance

# %%
image = np.zeros((100, 100))
t = np.linspace(0, 3, 20)
traj_x = np.where(
    t < 1.6, (40 * (np.cos(t) + 1)), 40 * (np.cos(t + 0.3) + 1) + 10 * (t - 1)
).astype(int)
traj_y = np.where(
    t < 1.6, (40 * (np.sin(t) + 1)), 40 * (np.sin(t + 0.3) + 1) + 10 * (t - 1)
).astype(int)
image[traj_y, traj_x] = 1
plt.imshow(image, cmap="gray")
# %%
bezier_coeffs = np.array([(3, 3), (5, 25), (60, 40), (90, 90)]).T
traj = interpolate_bezier(bezier_coeffs, t)
x, y = traj.astype(int).clip(0, 100)
plt.imshow(image, cmap=plt.cm.colors.ListedColormap(["black", "blue", "red"]))
plt.plot(x, y)


# %%
traj_fit = interpolate_bezier(fit_bezier((traj_x, traj_y), t), t)
x, y = traj_fit
plt.imshow(image, cmap="gray")
plt.plot(x, y)
# %%
std_distance(traj, traj_fit)
