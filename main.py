import net_training as nt

datadir = "D:\\inzynierka\\data\\"
imagedir = "cohn-kanade-images\\"
labeldir = "Emotion\\"

network = nt.build_cnn('model378.npz')

nt.train_net(datadir, imagedir, labeldir, network)

faces = nt.load_img("fear.jpg")
tab = nt.evaluate(network, faces)
print ("anger = {:.2f}%\ncontempt = {:.2f}%\ndisgust = {:.2f}%\nfear = {:.2f}%\nhappy = {:.2f}%\nsadness = {:.2f}%\nsurprise = {:.2f}%\n".format(tab[0][0]*100,tab[0][1]*100,tab[0][2]*100,tab[0][3]*100,tab[0][4]*100,tab[0][5]*100,tab[0][6]*100))

