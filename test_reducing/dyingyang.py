import numpy as np
import matplotlib.pyplot as plt


# Using the adapted YinYangDataset class
class YinYangDataset:
    def __init__(self, r_small=0.1, r_big=0.5, size=500, seed=42, transform=None):
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']

        for i in range(size):
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            x_flipped = 1.0 - x
            y_flipped = 1.0 - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        found_sample_yet = False
        while not found_sample_yet:
            x, y = self.rng.rand(2) * 2.0 * self.r_big
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
        return x, y, c

    def which_class(self, x, y):
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        return 2 if is_circles else int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.__cs)


# Generate YinYang dataset with 500 points
dataset = YinYangDataset(size=1000)

# Extract points and classes for plotting
points = np.array([dataset[i][0][:2] for i in range(len(dataset))])  # Use original x, y points only
classes = np.array([dataset[i][1] for i in range(len(dataset))])

# Define colors
colors = ['blue', 'orange', 'red']  # yin, yang, dot

# Plot the YinYang pattern
plt.figure(figsize=(6, 6))
for c in np.unique(classes):
    plt.scatter(points[classes == c, 0], points[classes == c, 1], label=dataset.class_names[c], color=colors[c], s=10)

plt.title("Yin-Yang Dataset Sample")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.show()
