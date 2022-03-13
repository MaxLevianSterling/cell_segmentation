# Get image coordinates when image is embedded in black
true_points = np.argwhere(values[0])
bottom_right = true_points.max(axis=0)

# Original encoding act_fn
act_fn = nn.LeakyReLU(0.2, inplace=True)