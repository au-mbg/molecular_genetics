import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def make_colors(n_children=2):
    # Make a colormap between green and red with n_children steps
    if n_children > 1:
        base_colors = ['green', 'yellow', 'red']
    else:
        base_colors = ['green', 'red']

    cmap = LinearSegmentedColormap.from_list('green_red', base_colors, N=n_children + 1)
    color_list = []

    for i in range(n_children + 1):
        color_list.append(cmap(i))

    return color_list





def simulate_disease_inheritance_mc(n_children, n_samples):
    rng = np.random.default_rng()
    sample_raw = rng.uniform(low=0, high=1, size=(n_samples, n_children))
    return sample_raw


def simulate_one_child_inheritance_mc(n_samples: int, inheritance_probability: float = 0.5):


    # Monte Carlo simulation
    n_children = 1
    sample_raw_mc = simulate_disease_inheritance_mc(n_children, n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 

    # Bar plot of random segregation values
    ax = axes[0]

    colors = ['red' if x <= inheritance_probability else 'green' for x in sample_raw_mc[:, 0]]
    ax.bar(sample_raw_mc[:, 0], np.ones_like(sample_raw_mc[:, 0]), width=0.01, alpha=0.7, edgecolor='black', color=colors)


    ax.set_xlabel('Child random segregation value')
    ax.set_yticks([])
    ax.set_title('Monte Carlo Simulation of Disease Inheritance')
    ax.axvline(inheritance_probability, color='red', linestyle='--')
    ax.text(inheritance_probability + 0.02, 1.05, 'Inheritance \n Threshold', color='red', rotation=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    # Histogram of results
    ax = axes[1]
    inherited_count = np.sum(sample_raw_mc[:, 0] <= inheritance_probability)
    not_inherited_count = n_samples - inherited_count

    ax.bar(['Does Not Inherit', 'Inherits Disease'], [not_inherited_count, inherited_count], color=['green', 'red'], edgecolor='black', alpha=0.7)
    ax.text(0, not_inherited_count - n_samples * 0.02, str(not_inherited_count), ha='center', va='top', fontsize=12)
    ax.text(1, inherited_count - n_samples * 0.02, str(inherited_count), ha='center', va='top', fontsize=12)

    ax.set_ylabel('Number of Samples')
    plt.show()

def one_child_convergence(inheritance_probability: float = 0.5): 
    n_samples = [10, 50, 100, 1000, 5000, 10000]
    trials = 500
    fractions = np.zeros((len(n_samples), trials))

    for i, n_sample in enumerate(n_samples):
        for t in range(trials):
            sample = simulate_disease_inheritance_mc(1, n_sample)
            fractions[i, t] = (sample < inheritance_probability).sum() / n_sample

    fig, ax = plt.subplots()
    ax.boxplot(fractions.T)

    ax.set_xticks(np.arange(1, len(n_samples)+1))
    ax.set_xticklabels(n_samples)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Fraction of inheritance')

    plt.show()

    
def simulate_two_children_inheritance_mc(n_samples: int, inheritance_probability: float = 0.5):

    def get_color(x, y, threshold=inheritance_probability):
        if x <= threshold and y <= threshold:
            return 'red'  # Both inherit disease
        elif x > threshold and y <= threshold:
            return 'orange'  # Only child 1 inherits
        elif x <= threshold and y > threshold:
            return 'yellow'  # Only child 2 inherits
        else:
            return 'green'  # Neither inherits disease

    # Monte Carlo simulation
    n_children = 2
    sample_raw_mc = simulate_disease_inheritance_mc(n_children, n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 

    colors = [get_color(x, y, threshold=inheritance_probability) for x, y in sample_raw_mc]


    ax = axes[0]
    ax.scatter(sample_raw_mc[:, 0], sample_raw_mc[:, 1], alpha=0.7, c=colors, edgecolors='black')

    ax.set_xlabel('Child 1 random segregation value')
    ax.set_ylabel('Child 2 random segregation value')
    ax.set_title('Monte Carlo Simulation of Disease Inheritance')
    ax.axvline(inheritance_probability, color='red', linestyle='--')
    ax.axhline(inheritance_probability, color='red', linestyle='--')

    text_box_settings = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.8)
    ax.text(0.025, 0.025, 'Both Inherit Disease', color='red', va='bottom', ha='left', bbox=text_box_settings)
    ax.text(0.975, 0.025, 'Only Child 1 Inherits', color='red', va='bottom', ha='right', bbox=text_box_settings)
    ax.text(0.025, 0.975, 'Only Child 2 Inherits', color='red', va='top', ha='left', bbox=text_box_settings)
    ax.text(0.975, 0.975, 'Neither Inherits Disease', color='red', va='top', ha='right', bbox=text_box_settings)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Histogram of results
    ax = axes[1]
    counts = {color: colors.count(color) for color in set(colors)}

    for color, count in counts.items():
        if color == 'green':
            ax.bar(0, count, color=color, edgecolor='black', alpha=0.7)
        elif color == 'orange':
            ax.bar(1, count, color=color, edgecolor='black', alpha=0.7)
        elif color == 'yellow':
            ax.bar(1, count, color=color, edgecolor='black', alpha=0.7, bottom=counts.get('orange', 0))
        elif color == 'red':
            ax.bar(2, count, color=color, edgecolor='black', alpha=0.7)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['None inherit', 'One inherits', 'Both inherit'])

    ax.set_xlabel('Inheritance Outcome')
    ax.set_ylabel('Number of Samples')
    plt.show()
