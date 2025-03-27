import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

# which directory to store model weights
WEIGHT_DIR = "./weights"
# style image (e.g. a painting)
STYLE_PATH = "./images/style/flint.jpg"
# target image, to which we apply the style
TARGET_PATH = "./images/target/tree.jpg"
# where to save the new image
SAVE_PATH = "./results/tree_flint.png"

# hyper parameters
NUM_EPOCHS = 5000
LR = 0.01
CONTENT_COEFFICIENT = 1.0
STYLE_COEFFICIENT = 1e6

# whether to show loss curves and preprocessed images
DETAIL = True

# define which layers to look at inside vgg16
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYERS = ['conv4_2']