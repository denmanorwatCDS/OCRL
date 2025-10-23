import numpy as np
from sklearn import decomposition

def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors

def get_option_colors(options, color_range=4):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
        option_colors = get_2d_colors(options, (-color_range, -color_range), (color_range, color_range))
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors

def monte_carlo_value_difference(rewards, gamma):
    # Returns monte-carlo estimate of discounted sum of rewards
    # between current and last state of supplied trajectories
    value_difference = np.zeros(rewards.shape)
    # Omit last reward as it is for performing action from last state.
    for i in reversed(range(rewards.shape[1] - 1)):
        value_difference[:, :i] = value_difference[:, :i] * gamma
        value_difference[:, :i] = value_difference[:, :i] + rewards[:, :i]
    return value_difference

class StatisticsCalculator():
    def __init__(self, name):
        self.name = name
        self.dict_buffer = {}
    
    def save_iter(self, logs_dict):
        for key in logs_dict:
            if key not in self.dict_buffer:
                self.dict_buffer[key] = []
            if isinstance(logs_dict[key], (float, int)):
                self.dict_buffer[key].append(logs_dict[key])
            else:    
                self.dict_buffer[key].append(logs_dict[key].item())
    
    def pop_statistics(self):
        outp = {}
        for key in self.dict_buffer:
            q1 = np.quantile(self.dict_buffer[key], q = 0.25)
            q2 = np.quantile(self.dict_buffer[key], q = 0.5)
            q3 = np.quantile(self.dict_buffer[key], q = 0.75)
            outp[self.name + '/' + key] = q2

        self.dict_buffer = {}
        return outp