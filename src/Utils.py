import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from enum import Enum
import random
# ================================================================================================================
# Classes
# ================================================================================================================
class SEABORN_COLORMAPS(Enum):
    ACCENT = "Accent"
    ACCENT_R = "Accent_r"
    AFMHOT = "afmhot"
    AFMHOT_R = "afmhot_r"
    AUTUMN = "autumn"
    AUTUMN_R = "autumn_r"
    BINARY = "binary"
    BINARY_R = "binary_r"
    BLUES = "Blues"
    BLUES_R = "Blues_r"
    BONE = "bone"
    BONE_R = "bone_r"
    BRBG = "BrBG"
    BRBG_R = "BrBG_r"
    BRG = "brg"
    BRG_R = "brg_r"
    BWR = "bwr"
    BWR_R = "bwr_r"
    BUGN = "BuGn"
    BUGN_R = "BuGn_r"
    BUPU = "BuPu"
    BUPU_R = "BuPu_r"
    CMRMAP = "CMRmap"
    CMRMAP_R = "CMRmap_r"
    CIVIDIS = "cividis"
    CIVIDIS_R = "cividis_r"
    COOL = "cool"
    COOL_R = "cool_r"
    COOLWARM = "coolwarm"
    COOLWARM_R = "coolwarm_r"
    COPPER = "copper"
    COPPER_R = "copper_r"
    CREST = "crest"
    CREST_R = "crest_r"
    CUBEHELIX = "cubehelix"
    CUBEHELIX_R = "cubehelix_r"
    DARK2 = "Dark2"
    DARK2_R = "Dark2_r"
    FLAG = "flag"
    FLAG_R = "flag_r"
    FLARE = "flare"
    FLARE_R = "flare_r"
    GIST_EARTH = "gist_earth"
    GIST_EARTH_R = "gist_earth_r"
    GIST_GRAY = "gist_gray"
    GIST_GRAY_R = "gist_gray_r"
    GIST_GREY = "gist_grey"
    GIST_HEAT = "gist_heat"
    GIST_HEAT_R = "gist_heat_r"
    GIST_NCAR = "gist_ncar"
    GIST_NCAR_R = "gist_ncar_r"
    GIST_RAINBOW = "gist_rainbow"
    GIST_RAINBOW_R = "gist_rainbow_r"
    GIST_STERN = "gist_stern"
    GIST_STERN_R = "gist_stern_r"
    GIST_YARG = "gist_yarg"
    GIST_YARG_R = "gist_yarg_r"
    GNU_PLOT = "gnuplot"
    GNU_PLOT_R = "gnuplot_r"
    GNU_PLOT2 = "gnuplot2"
    GNU_PLOT2_R = "gnuplot2_r"
    GRAY = "gray"
    GRAY_R = "gray_r"
    GREENS = "Greens"
    GREENS_R = "Greens_r"
    GREYS = "Greys"
    GREYS_R = "Greys_r"
    HOT = "hot"
    HOT_R = "hot_r"
    HSV = "hsv"
    HSV_R = "hsv_r"
    ICEFIRE = "icefire"
    ICEFIRE_R = "icefire_r"
    INFERNO = "inferno"
    INFERNO_R = "inferno_r"
    JET = "jet"
    JET_R = "jet_r"
    MAGMA = "magma"
    MAGMA_R = "magma_r"
    MAKO = "mako"
    MAKO_R = "mako_r"
    NIPY_SPECTRAL = "nipy_spectral"
    NIPY_SPECTRAL_R = "nipy_spectral_r"
    OCEAN = "ocean"
    OCEAN_R = "ocean_r"
    ORANGES = "Oranges"
    ORANGES_R = "Oranges_r"
    ORRD = "OrRd"
    ORRD_R = "OrRd_r"
    PAIRED = "Paired"
    PAIRED_R = "Paired_r"
    PASTEL1 = "Pastel1"
    PASTEL1_R = "Pastel1_r"
    PASTEL2 = "Pastel2"
    PASTEL2_R = "Pastel2_r"
    PINK = "pink"
    PINK_R = "pink_r"
    PIYG = "PiYG"
    PIYG_R = "PiYG_r"
    PLASMA = "plasma"
    PLASMA_R = "plasma_r"
    PRGN = "PRGn"
    PRGN_R = "PRGn_r"
    PRISM = "prism"
    PRISM_R = "prism_r"
    PUBU = "PuBu"
    PUBU_R = "PuBu_r"
    PUBU_GN = "PuBuGn"
    PUBU_GN_R = "PuBuGn_r"
    PUOR = "PuOr"
    PUOR_R = "PuOr_r"
    PURPLES = "Purples"
    PURPLES_R = "Purples_r"
    PURD = "PuRd"
    PURD_R = "PuRd_r"
    RAINBOW = "rainbow"
    RAINBOW_R = "rainbow_r"
    RDBU = "RdBu"
    RDBU_R = "RdBu_r"
    RDGY = "RdGy"
    RDGY_R = "RdGy_r"
    RDPU = "RdPu"
    RDPU_R = "RdPu_r"
    RDYLBU = "RdYlBu"
    RDYLBU_R = "RdYlBu_r"
    RDYLGN = "RdYlGn"
    RDYLGN_R = "RdYlGn_r"
    REDS = "Reds"
    REDS_R = "Reds_r"
    ROCKET = "rocket"
    ROCKET_R = "rocket_r"
    SEISMIC = "seismic"
    SEISMIC_R = "seismic_r"
    SET1 = "Set1"
    SET1_R = "Set1_r"
    SET2 = "Set2"
    SET2_R = "Set2_r"
    SET3 = "Set3"
    SET3_R = "Set3_r"
    SPECTRAL = "Spectral"
    SPECTRAL_R = "Spectral_r"
    SPRING = "spring"
    SPRING_R = "spring_r"
    SUMMER = "summer"
    SUMMER_R = "summer_r"
    TAB10 = "tab10"
    TAB10_R = "tab10_r"
    TAB20 = "tab20"
    TAB20_R = "tab20_r"
    TAB20B = "tab20b"
    TAB20B_R = "tab20b_r"
    TAB20C = "tab20c"
    TAB20C_R = "tab20c_r"
    TERRAIN = "terrain"
    TERRAIN_R = "terrain_r"
    TURBO = "turbo"
    TURBO_R = "turbo_r"
    TWILIGHT = "twilight"
    TWILIGHT_R = "twilight_r"
    TWILIGHT_SHIFTED = "twilight_shifted"
    TWILIGHT_SHIFTED_R = "twilight_shifted_r"
    VIRIDIS = "viridis"
    VIRIDIS_R = "viridis_r"
    VLAG = "vlag"
    VLAG_R = "vlag_r"
    WISTIA = "Wistia"
    WISTIA_R = "Wistia_r"
    WINTER = "winter"
    WINTER_R = "winter_r"
    YLGN = "YlGn"
    YLGNBU = "YlGnBu"
    YLGNBU_R = "YlGnBu_r"
    YLGN_R = "YlGn_r"
    YLORBR = "YlOrBr"
    YLORBR_R = "YlOrBr_r"
    YLORRD = "YlOrRd"
    YLORRD_R = "YlOrRd_r"
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
# Seaborne
# ================================================================================================================
def plot_confusion_matrix(y_true, y_pred, cmap=SEABORN_COLORMAPS.BLUES.value):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                xticklabels=["Positive", "Negative"],
                yticklabels=["Positive", "Negative"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def plot_ROC(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
