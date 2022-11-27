import random
import torch


manualSeed = 9999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataroot = "/app/sbml/dcgan/project_data/102flowers"


# dataroot = '/app/sbml/hello_gan/project_data/CUB_200_2011/images'
# dataset_name = 'birds'

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128 # 250 # 64 244

# Number of channels in the training images. For color images this is 3
nc = 3
batch_size = 128

dataset_name = f'test_{image_size}_batch{batch_size}'

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

workers = 2


# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sub directory for data
sample_dir = f"samples_{dataset_name}"
print(f"sample_dir={sample_dir}")