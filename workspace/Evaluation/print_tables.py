import json
import tensorflow as tf


MEANS = True


with open('../Results/auroc_aupr.json') as json_file:
    data = json.load(json_file)


def make_latex_table(metric, mean=True):
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
            index = 0 if h == "effnetb3" else 2
            values.append(round(function(data[method][h][metric][index], axis=-1).numpy(), 3))
        #out = [str(val) + " $\pm$ " + str(std) for val, std in zip(values, stddevs)]
        texdata += f"{title} & {' & '.join(map(str, values))} \\\\\n"

    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata, end="")
    print("\\end{tabular}")


print("---------------------------------------AUROC------------------------------------------")
make_latex_table("auroc", MEANS)
print("----------------------------------------AUPR------------------------------------------")
make_latex_table("aupr", MEANS)
print("\n\n\n")

keys = ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]


def nuc_runtimes(nuc):
    output = "Tr."
    for key in keys:
        output += " & "
        output += str(round(
            tf.reduce_mean(t[key][nuc]["preparation & uncertainty"]).numpy() -
            tf.reduce_mean(t[key][nuc]["uncertainty"]).numpy(), 1))
    output += " //"
    print(output)
    output = "CEs"
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][nuc]["uncertainty"]).numpy(), 1))
    output += " //"
    print(output)


def ensemble_runtimes(ens):
    output = "Tr."
    for key in keys:
        output += " & "
        times = tf.reduce_mean(t[key][ens]["preparation & uncertainty"]) - tf.reduce_mean(t[key][ens]["uncertainty"]) +\
                tf.reduce_mean(t[key][ens]["preparation & calibration"]) - tf.reduce_mean(t[key][ens]["with calibration"])
        output += str(round(times.numpy()*0.5, 1))
    output += " //"
    print(output)
    output = "CEs"
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][ens]["uncertainty"]).numpy(), 1))
    output += " //"
    print(output)
    output = "kal."
    for key in keys:
        output += " & "
        output += str(round(tf.reduce_mean(t[key][ens]["with calibration"]).numpy(), 1))
    output += " //"
    print(output)


with open('../Results/times.json') as json_file:
    t = json.load(json_file)

print("--------------------------------------TIMES--------------------------------------\n")
print("& \multicolumn{6}{c}{SOFTMAX SE} \\ \midrule")
output = "unk."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["Softmax SE"]["uncertainty"]).numpy(), 3))
output += " //"
print(output)
output = "kal."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["Softmax SE"]["calibration"]).numpy(), 3))
output += " //"
print(output)

print("\midrule & \multicolumn{6}{c}{MONTE CARLO DROPOUT} \\ \midrule")
output = "unk."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["MC Dropout"]["uncertainty"]).numpy(), 1))
output += " //"
print(output)
output = "kal."
for key in keys:
    output += " & "
    output += str(round(tf.reduce_mean(t[key]["MC Dropout"]["with calibration"]).numpy(), 1))
output += " //"
print(output)

print("\midrule & \multicolumn{6}{c}{ENSEMBLES} \\ \midrule")
print("& \multicolumn{6}{c}{Bagging} \\")
print("\ arrayrulecolor{gray} \midrule")
ensemble_runtimes("Bagging")

print("\midrule & \multicolumn{6}{c}{\textit{Zufällige Initialisierung und Shuffling der Daten}} \\ \midrule")
ensemble_runtimes("ZIS")

print("\midrule & \multicolumn{6}{c}{\textit{Data Augmentation}} \\ \midrule")
ensemble_runtimes("Data Augmentation")

print("\ arrayrulecolor{black} \midrule")
print("& \multicolumn{6}{c}{NUC TRAINING} \\ \midrule")
nuc_runtimes("NUC Training")

print("\midrule & \multicolumn{6}{c}{NUC VALIDATION} \\ \midrule")
nuc_runtimes("NUC Validation")


print("\n\n\n")
print("------------------------------------------ECES----------------------------------------------")
print("\n\n")


def ece_row(title):
    text = title
    for key in ["CNN_cifar10_100", "CNN_cifar10_1000", "CNN_cifar10_10000", "CNN_cifar10", "CNN_cifar100", "effnetb3"]:
        text += " & "
        text += str(round(tf.reduce_mean(ece[key][title]).numpy(), 3))
    text += " \\"
    print(text)


with open('../Results/eces.json') as json_file:
    ece = json.load(json_file)

print("& CNN c$10_{100}$ & CNN c$10_{1000}$ & CNN c$10_{10000}$ & CNN c10 & CNN c100 & EffNet \\")
print("\midrule")
ece_row("Soft SE")
print("\midrule")
ece_row("MCD SE")
ece_row("MCD MI")
print("\midrule")
ece_row("Bag SE")
ece_row("Bag MI")
ece_row("ZIS SE")
ece_row("ZIS MI")
ece_row("DA SE")
ece_row("DA MI")
print("\midrule")
ece_row("NUC Tr")
ece_row("NUC Va")