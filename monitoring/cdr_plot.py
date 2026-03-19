import matplotlib.pyplot as plt

def plot_cdr(history):

    plt.plot(history)
    plt.title("CDR Progression")
    plt.xlabel("Visit")
    plt.ylabel("CDR")

    plt.savefig("cdr_progression.png")