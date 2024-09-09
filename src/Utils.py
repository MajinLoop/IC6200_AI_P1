import matplotlib.pyplot as plt
from enum import Enum
import random
# ================================================================================================================
# Classes
# ================================================================================================================

# ================================================================================================================
# general
# ================================================================================================================
def printt(name, value):
    print(f"{name} = {value}, type = {type(value)}")
# ================================================================================================================
# Matplotlib
# ================================================================================================================
def plt_bar(categories,
            magnitudes,
            fig_width=10,
            fig_height=6,
            color="blue",
            title="Title",
            title_fontsize=16,
            x_label="X axis",
            x_fontsize=14,
            xticks_labels=[],
            y_label="Y axis",
            y_fontsize=14):
    """
    Generate and shows a matplotlib bar type plot.

    Returns
    -------
    None
    """
    # Create a vertical bar chart
    plt.figure(figsize=(fig_width, fig_height))
    plt.bar(categories, magnitudes, color=color)

    # Adding titles and labels
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x_label, fontsize=x_fontsize)
    plt.ylabel(y_label, fontsize=y_fontsize)

    if(xticks_labels != []):
        plt.xticks(ticks=categories, labels=xticks_labels)

    # Show the plot
    plt.show()
# ================================================================================================================
# Pandas
# ================================================================================================================
