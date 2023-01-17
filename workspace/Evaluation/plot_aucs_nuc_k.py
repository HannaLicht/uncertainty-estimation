import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions
x = [5, 10, 25, 50, 100]
valid = True

# AUROCs for nuc trained on validation data
c10_100_val_roc = [0.73482597, 0.746, 0.75602734, 0.762831, 0.76554745]
c10_1000_val_roc = [0.7392374, 0.746, 0.7495748, 0.75054413, 0.7502146]
c10_10000_val_roc = [0.780, 0.779, 0.773, 0.772, 0.769]
c10_val_roc = [0.825, 0.824, 0.808, 0.804, 0.805]
c100_val_roc = [0.80399644, 0.803, 0.7999729, 0.7871314, 0.7688228]
effnet_val_roc = [0.706, 0.676, 0.663, 0.622, 0.626]

# AUPRs for nuc trained on validation data
c10_100_val_pr = [0.58416265, 0.601, 0.60948586, 0.6230147, 0.62302554]
c10_1000_val_pr = [0.7557357, 0.764, 0.7682079, 0.77033293, 0.7690269]
c10_10000_val_pr = [0.867, 0.865, 0.861, 0.863, 0.862]
c10_val_pr = [0.926, 0.926, 0.918, 0.917, 0.918]
c100_val_pr = [0.7694196, 0.768, 0.7638339, 0.7492295, 0.7310469]
effnet_val_pr = [0.778, 0.758, 0.747, 0.716, 0.713]

# AUROCs for nuc trained on train data
c10_100_tra_roc = [0.5741764, 0.51731193, 0.50154805, 0.47292584, 0.48320332]
c10_1000_tra_roc = [0.6989664, 0.70758396, 0.7129875, 0.6906346, 0.69620407]
c10_10000_tra_roc = [0.776418, 0.76054287, 0.7662318, 0.7655159, 0.75060755]
c10_tra_roc = [0.8265209, 0.82567924, 0.8231063, 0.81378764, 0.8161671]
c100_tra_roc = [0.8117564, 0.8016536, 0.80501044, 0.7990899, 0.80300176]
effnet_tra_roc = [0.5020281, 0.50199217, 0.5021752, 0.50230366, 0.5015366]

# AUPRs for nuc trained on train data
c10_100_tra_pr = [0.39861143, 0.34708658, 0.33656523, 0.34436908, 0.33882308]
c10_1000_tra_pr = [0.7015376, 0.71906716, 0.72727036, 0.71054435, 0.7120254]
c10_10000_tra_pr = [0.86312705, 0.8488414, 0.8522699, 0.8527452, 0.8444082]
c10_tra_pr = [0.9274173, 0.9265839, 0.92510617, 0.92114574, 0.91920006]
c100_tra_pr = [0.7813339, 0.77127266, 0.7724488, 0.7645804, 0.7666045]
effnet_tra_pr = [0.6211535, 0.6211366, 0.62122315, 0.6212839, 0.62092143]


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_curves(auroc, valid):
    plt.xlabel("Anzahl Nachbarn (k)")
    plt.xticks([5, 10, 25, 50, 100])
    #plt.ylim((0.55, 0.95))

    if auroc:
        plt.ylabel("AUROC")
        if valid:
            aucs = [c10_100_val_roc, c10_1000_val_roc, c10_10000_val_roc, c10_val_roc, c100_val_roc, effnet_val_roc]
        else:
            aucs = [c10_100_tra_roc, c10_1000_tra_roc, c10_10000_tra_roc, c10_tra_roc, c100_tra_roc, effnet_tra_roc]

    else:
        plt.ylabel("AUPR")
        if valid:
            aucs = [c10_100_val_pr, c10_1000_val_pr, c10_10000_val_pr, c10_val_pr, c100_val_pr, effnet_val_pr]
        else:
            aucs = [c10_100_tra_pr, c10_1000_tra_pr, c10_10000_tra_pr, c10_tra_pr, c100_tra_pr, effnet_tra_pr]

    plt.plot(x, aucs[0], label="Cifar10 (100 Bilder)", marker='.', color=adjust_lightness('b', 1.6))
    plt.plot(x, aucs[1], label="Cifar10 (1000 Bilder)", marker='.', color=adjust_lightness('b', 1.3))
    plt.plot(x, aucs[2], label="Cifar10 (10000 Bilder)", marker='.', color=adjust_lightness('b', 0.8))
    plt.plot(x, aucs[3], label="Cifar10", marker='.', color=adjust_lightness('b', 0.4))
    plt.plot(x, aucs[4], label="Cifar100", marker='.', color='tomato')
    plt.plot(x, aucs[5], label="Cars196", marker='.', color='yellowgreen')


plt.figure(figsize=(9, 3.5))

plt.subplot(1, 2, 1)
plot_curves(True, valid)
box = plt.subplot(1, 2, 1).get_position()
#plt.subplot(1, 2, 1).set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

plt.subplot(1, 2, 2)
plot_curves(False, valid)
box = plt.subplot(1, 2, 2).get_position()
#plt.subplot(1, 2, 2).set_position([box.x0*0.9, box.y0, box.width * 0.8, box.height*0.8])

plt.subplots_adjust(left=0.09, right=0.75, bottom=0.16, top=0.9, wspace=0.28, hspace=0.35)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plot_name = '../plots/AUCs_nuc_different_k_valid.pdf' if valid else 'plots/AUCs_nuc_different_k_train.pdf'
plt.savefig(plot_name)
plt.show()