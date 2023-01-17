import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


x = [5, 10, 25, 50, 100]

c10_100 = [0.73482597, 0.7465769, 0.75602734, 0.762831, 0.76554745]
c10_1000 = [0.7392374, 0.7450949, 0.7495748, 0.75054413, 0.7502146]
c10_10000 = [0.780, 0.778, 0.773, 0.772, 0.769]
c10 = [0.825, 0.823, 0.808, 0.804, 0.805]
c100 = [0.80399644, 0.8025899, 0.7999729, 0.7871314, 0.7688228]
imgnet = []

plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.ylabel("AUROC")
plt.xlabel("Anzahl Nachbarn (k)")
plt.plot(x, c10_100, label="cifar10 (100 data)", marker='.', color=adjust_lightness('b', 1.6))
plt.plot(x, c10_1000, label="cifar10 (1000 data)", marker='.', color=adjust_lightness('b', 1.3))
plt.plot(x, c10_10000, label="cifar10 (10000 data)", marker='.', color=adjust_lightness('b', 0.8))
plt.plot(x, c10, label="cifar10", marker='.', color=adjust_lightness('b', 0.4))
plt.plot(x, c100, label="cifar100", marker='.', color='tomato')
#plt.plot(x, imgnet, label="imagenet", marker='.', color='yellowgreen')

box = plt.subplot(1, 2, 1).get_position()
plt.subplot(1, 2, 1).set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

c10_100 = [0.58416265, 0.602342, 0.60948586, 0.6230147, 0.62302554]
c10_1000 = [0.7557357, 0.76433206, 0.7682079, 0.77033293, 0.7690269]
c10_10000 = [0.867, 0.863, 0.861, 0.863, 0.862]
c10 = [0.926, 0.926, 0.918, 0.917, 0.918]
c100 = [0.7694196, 0.76708686, 0.7638339, 0.7492295, 0.7310469]
imgnet = []

plt.subplot(1, 2, 2)
plt.ylabel("AUPR")
plt.xlabel("Anzahl Nachbarn (k)")
plt.plot(x, c10_100, label="cifar10 (100 data)", marker='.', color=adjust_lightness('b', 1.6))
plt.plot(x, c10_1000, label="cifar10 (1000 data)", marker='.', color=adjust_lightness('b', 1.3))
plt.plot(x, c10_10000, label="cifar10 (10000 data)", marker='.', color=adjust_lightness('b', 0.8))
plt.plot(x, c10, label="cifar10", marker='.', color=adjust_lightness('b', 0.4))
plt.plot(x, c100, label="cifar100", marker='.', color='tomato')
#plt.plot(x, imgnet, label="imagenet", marker='.', color='yellowgreen')

box = plt.subplot(1, 2, 2).get_position()
plt.subplot(1, 2, 2).set_position([box.x0*0.9, box.y0, box.width * 0.8, box.height*0.8])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
