import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from PIL import Image
from skimage import io, transform
import matplotlib.pyplot as plt
from keras import metrics

trainNum = 39
testNum = 11
w, h = 50,60
X_train = np.zeros((trainNum*2, h*w))
X_test = np.zeros((testNum*2, h*w))
y_train = np.zeros((trainNum*2,1), np.int)
y_test = np.zeros((testNum*2,1), np.int)

# 讀圖& resize到60*50, 每張reshape成1*3000
for i in range(trainNum):
    X_train[i, :] = np.array(Image.open('D:\\Downloads\\DB\\Man(50)\\Training\\'+str(i+1)+'.jpg').convert('L').resize((w,h)),dtype=np.float64).reshape(1, -1)
    y_train[i] = 1
for i in range(trainNum):
    X_train[i+39, :] = np.array(Image.open('D:\\Downloads\\DB\\Woman(50)\\Training\\'+'W('+str(i+1)+')'+'.jpg').convert('L').resize((w,h)),dtype=np.float64).reshape(1, -1)
    y_train[i+39] = 0
for i in range(testNum):
    X_test[i, :] = np.array(Image.open('D:\\Downloads\\DB\\Man(50)\\Testing\\'+str(i+40)+'.jpg').convert('L').resize((w,h)),dtype=np.float).reshape(1, -1)
    y_test[i] = 1
for i in range(testNum):
    X_test[i+11, :] = np.array(Image.open('D:\\Downloads\\DB\\Woman(50)\\Testing\\'+'W('+str(i+40)+')'+'.jpg').convert('L').resize((w,h)),dtype=np.float).reshape(1, -1)
    y_test[i+11] = 0
    
# convert y to one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# select optimizer
optim = 'adam'

# dimension of latent space
n_z = 5

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]

# nubmer of epochs, batch size
m = 5
n_epoch = 70

##  ENCODER ##

# encoder inputs
X = Input(shape=(w*h, ))
cond = Input(shape=(n_y, ))

# merge pixel representation and label
inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
l1 = Dense(1500, activation='relu')(inputs)
l2 = Dense(750, activation='relu')(l1)
l3 = Dense(300, activation='relu')(l2)
mu = Dense(n_z, activation='linear')(l3)
log_sigma = Dense(n_z, activation='linear')(l3)

def sample_z(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_z), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##  DECODER  ##

# dense ReLU to sigmoid layers
decoder_hidden = Dense(300, activation='relu')
decoder_hidden1 = Dense(750, activation='relu')
decoder_hidden2 = Dense(1500, activation='relu')
decoder_out = Dense(3000, activation='sigmoid')
d1 = decoder_hidden(z_cond)
d2 = decoder_hidden1(d1)
d3 = decoder_hidden2(d2)
outputs = decoder_out(d3)

# define cvae models
cvae = Model([X, cond], outputs)

# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = w* h* metrics.binary_crossentropy(y_true, y_pred)
    # D_KL(Q(z|X) || P(z|X))
    kl =-0.5 * K.sum(1 + log_sigma - K.exp(log_sigma) - K.square(mu), axis=-1)
    #return recon + kl
    return K.mean(recon + kl)

def KL_loss(y_true, y_pred):
	return(-0.5 * K.sum(1 + log_sigma - K.exp(log_sigma) - K.square(mu), axis=-1))

def recon_loss(y_true, y_pred):
	#return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))
    return( w* h* metrics.binary_crossentropy(y_true, y_pred))


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])
cvae_hist = cvae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test))

# input testpic in man's and woman's labels
    
testp_y_man = np.zeros((1,2))
testp_y_woman = np.zeros((1,2))
#testp = io.imread("output.jpg", dtype = np.float64)
testp = io.imread("26.jpg", dtype = np.float64)
testp = transform.resize(testp, (60,50,1))
testp = testp.astype('float32') /255.
testp = testp.reshape(1,-1)
x_ori = testp
testp_y_man[0,1] = 1
testp_y_woman[0,0] = 1
cvae_img_man = cvae.predict([testp,testp_y_man])
cvae_img_woman = cvae.predict([testp,testp_y_woman])
plt.subplot(1,3, 1)
plt.title('original')
plt.imshow(x_ori.reshape(60, 50), cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1,3, 2)
plt.title('cvae_man')
plt.imshow(cvae_img_man.reshape(60, 50), cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1,3, 3)
plt.title('cvae_woman')
plt.imshow(cvae_img_woman.reshape(60, 50), cmap=plt.cm.gray)
plt.axis('off')

