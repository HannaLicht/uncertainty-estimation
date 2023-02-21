import json
import tensorflow as tf


def make_latex_table(metric, mean=True):
    with open('../Results/auroc_aupr.json') as json_file:
        data = json.load(json_file)

    headers = data["MCdrop SE"].keys()
    titles = ["NUC Tr", "NUC Va", "Soft SE", "MCD SE", "MCD MI", "Bag SE", "Bag MI", "ZIS SE", "ZIS MI", "DA SE", "DA MI"]
    function = tf.reduce_mean if mean else tf.math.reduce_std

    textabular = f"l|{'r' * len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\ \midrule \n"

    for count, (method, title) in enumerate(zip(data.keys(), titles)):
        if count < 2:
            continue
        if count == 3 or count == 5:
            texdata += "\midrule \n"
        values = [round(function(data[method][m][metric], axis=-1).numpy(), 3) for m in headers]
        #out = [str(val) + " $\pm$ " + str(std) for val, std in zip(values, stddevs)]
        texdata += f"{title} & {' & '.join(map(str,values))} \\\\\n"

    texdata += "\midrule \n"
    for count, (method, title) in enumerate(zip(data.keys(), titles)):
        if count > 1:
            continue
        values = []
        for h in headers:
            if h == "CNN_cifar10_100" and title == "NUC Va":
                values.append(-1)
            else:
                values.append(round(function(data[method][h][metric][0]).numpy(), 3))
        texdata += f"{title} & {' & '.join(map(str, values))} \\\\\n"

    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata, end="")
    print("\\end{tabular}")


def ood_table(mean=True):
    with open('../Results/ood_auroc_aupr.json') as json_file:
        data = json.load(json_file)

    headers = ["cifar10", "cifar100", "cars196"]
    subheaders = ["auroc", "aupr", "auroc", "aupr", "auroc", "aupr"]
    titles = ["MCD SE", "MCD MI", "Bag SE", "Bag MI", "DA SE", "DA MI", "NUC Tr", "NUC Va", "Soft SE", "Max Soft"]
    function = tf.reduce_mean if mean else tf.math.reduce_std

    textabular = f"l|{'r' * len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texsubheader = " & " + " & ".join(subheaders) + "\\\\"
    texdata = "\\ \midrule \n"

    for count, title in enumerate(titles):
            if count == 2 or count == 6 or count == 8:
                texdata += "\midrule \n"
            values = []
            for dataset in headers:
                values.append(round(function(data[dataset][title]["auroc"], axis=-1).numpy(), 3))
                values.append(round(function(data[dataset][title]["aupr"], axis=-1).numpy(), 3))
            # out = [str(val) + " $\pm$ " + str(std) for val, std in zip(values, stddevs)]
            texdata += f"{title} & {' & '.join(map(str, values))} \\\\\n"

    print("\\begin{tabular}{" + textabular + "}")
    print(texheader)
    print(texsubheader)
    print(texdata, end="")
    print("\\end{tabular}")


print("---------------------------------------AUROC------------------------------------------")
make_latex_table("auroc", True)
print("----------------------------------------AUPR------------------------------------------")
make_latex_table("aupr", True)
print("\n\n\n")

print("---------------------------------------AUROC-STDDEV------------------------------------------")
make_latex_table("auroc", False)
print("----------------------------------------AUPR-STDEV------------------------------------------")
make_latex_table("aupr", False)
print("\n\n\n")


print("---------------------------------------AUROC-AUPR-OOD------------------------------------------")
ood_table(True)
print("\n\n\n")


keys = ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]


def nuc_runtimes(nuc):
    output = "Tr."
    for key in keys:
        output += " & "
        output += str(round(
            tf.reduce_mean(t[key][nuc]["preparation & uncertainty"]).numpy() -
            tf.reduce_mean(t[key][nuc]["uncertainty"]).numpy(), 1))
    output += " \\\\"
    print(output)
    output = "CEs"
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][nuc]["uncertainty"]).numpy(), 1))
    output += " \\\\"
    print(output)


def ensemble_runtimes(ens):
    output = "Tr."
    for key in keys:
        output += " & "
        times = tf.reduce_mean(t[key][ens]["preparation & uncertainty"]) - tf.reduce_mean(t[key][ens]["uncertainty"]) +\
                tf.reduce_mean(t[key][ens]["preparation & calibration"]) - tf.reduce_mean(t[key][ens]["with calibration"])
        output += str(round(times.numpy()*0.5, 1))
    output += " \\\\"
    print(output)
    output = "CEs"
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][ens]["uncertainty"]).numpy(), 1))
    output += " \\\\"
    print(output)
    output = "kal."
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][ens]["with calibration"]).numpy(), 1))
    output += " \\\\"
    print(output)


with open('../Results/times.json') as json_file:
    t = json.load(json_file)

print("--------------------------------------TIMES--------------------------------------\n")
print("& \multicolumn{6}{c}{SOFTMAX SE} \\\\ \midrule")
output = "unk."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["Softmax SE"]["uncertainty"]).numpy(), 3))
output += " \\\\"
print(output)
output = "kal."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["Softmax SE"]["calibration"]).numpy(), 3))
output += " \\\\"
print(output)

print("\midrule & \multicolumn{6}{c}{MONTE CARLO DROPOUT} \\\\ \midrule")
output = "unk."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["MC Dropout"]["uncertainty"]).numpy(), 1))
output += " \\\\"
print(output)
output = "kal."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["MC Dropout"]["with calibration"]).numpy(), 1))
output += " \\\\"
print(output)

print("\midrule & \multicolumn{6}{c}{ENSEMBLES} \\\\ \midrule")
print("& \multicolumn{6}{c}{Bagging} \\\\")
print("\ arrayrulecolor{gray} \midrule")
ensemble_runtimes("Bagging")

print("\midrule & \multicolumn{6}{c}{\textit{Zufällige Initialisierung und Shuffling der Daten}} \\\\ \midrule")
ensemble_runtimes("ZIS")

print("\midrule & \multicolumn{6}{c}{\textit{Data Augmentation}} \\\\ \midrule")
ensemble_runtimes("Data Augmentation")

print("\ arrayrulecolor{black} \midrule")
print("& \multicolumn{6}{c}{NUC TRAINING} \\\\ \midrule")
nuc_runtimes("NUC Tr")

print("\midrule & \multicolumn{6}{c}{NUC VALIDATION} \\\\ \midrule")
nuc_runtimes("NUC Va")


print("\n\n\n")
print("------------------------------------------ECES----------------------------------------------")
print("\n\n")


def ece_row(title, function):
    text = title
    for key in ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]:
        text += " & "
        text += str(round(function(ece[key][title]).numpy(), 3))
    text += " \\\\"
    print(text)


def ece_row_nuc(title, function):
    text = title
    for key in ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]:
        text += " & "
        try:
            text += str(round(function(ece[key][title][2]).numpy(), 3))
        except:
            text += "-"
    text += " \\\\"
    print(text)


with open('../Results/eces.json') as json_file:
    ece = json.load(json_file)

print("& CNN c$10_{100}$ & CNN c$10_{1000}$ & CNN c$10_{10000}$ & CNN c10 & CNN c100 & EffNet \\\\")
print("\midrule")
ece_row("Soft SE", tf.reduce_mean)
print("\midrule")
ece_row("MCD SE", tf.reduce_mean)
ece_row("MCD MI", tf.reduce_mean)
print("\midrule")
ece_row("Bag SE", tf.reduce_mean)
ece_row("Bag MI", tf.reduce_mean)
ece_row("DA SE", tf.reduce_mean)
ece_row("DA MI", tf.reduce_mean)
print("\midrule")
ece_row_nuc("NUC Tr", tf.reduce_mean)
ece_row_nuc("NUC Va", tf.reduce_mean)

print("\n\n\n")
print("------------------------------------------ECES-STDDEVS----------------------------------------------")
print("\n\n")

print("& CNN c$10_{100}$ & CNN c$10_{1000}$ & CNN c$10_{10000}$ & CNN c10 & CNN c100 & EffNet \\")
print("\midrule")
ece_row("Soft SE", tf.math.reduce_std)
print("\midrule")
ece_row("MCD SE", tf.math.reduce_std)
ece_row("MCD MI", tf.math.reduce_std)
print("\midrule")
ece_row("Bag SE", tf.math.reduce_std)
ece_row("Bag MI", tf.math.reduce_std)
ece_row("DA SE", tf.math.reduce_std)
ece_row("DA MI", tf.math.reduce_std)
print("\midrule")
ece_row_nuc("NUC Tr", tf.math.reduce_std)
ece_row_nuc("NUC Va", tf.math.reduce_std)