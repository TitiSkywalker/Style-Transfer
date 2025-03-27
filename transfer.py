import os 
import torch 
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import *
from vgg import vgg16 
from console import TaskLog, LoopLog
from image import preprocess_image, save_processed_image

def gram_matrix(feature):
    c, h, w = feature.size()
    features_reshaped = feature.view(c, h * w)
    gram = torch.mm(features_reshaped, features_reshaped.t())
    return gram.div(c * h * w)

class Manager:
    def __init__(self):
        with TaskLog("initialize Manager"):
            self.network = vgg16().to(device)

    def fetch_weights(self):
        with TaskLog("fetch vgg16 weights"):
            if not os.path.exists(WEIGHT_DIR):
                os.makedirs(WEIGHT_DIR)
            if os.path.exists(f"{WEIGHT_DIR}/vgg16.pth"):
                return
            torch_vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            torch.save(torch_vgg.state_dict(), f"{WEIGHT_DIR}/vgg16.pth")

    def load_weights(self):
        with TaskLog("load vgg16 weights"):
            self.network.load_state_dict(torch.load(f"{WEIGHT_DIR}/vgg16.pth", weights_only=True))

    def transfer(self):
        # preprocess style image and target(original) image
        with TaskLog(f"process style image {STYLE_PATH}"):
            style_image = preprocess_image(STYLE_PATH).to(device)
        with TaskLog(f"process target image {TARGET_PATH}"):
            original_image = preprocess_image(TARGET_PATH).to(device)

        # compute features of style image and original image
        with torch.no_grad():
            style_features = self.network.collect(style_image)
            style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}
            original_features = self.network.collect(original_image)

        target_image = original_image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target_image], lr=LR)
        loss = []

        with LoopLog("transfer style", count=NUM_EPOCHS, precise=True) as log:
            for epoch in range(NUM_EPOCHS):
                # compute current features
                target_features = self.network.collect(target_image)

                # compare it with original image
                content_loss = 0
                for layer in CONTENT_LAYERS:
                    content_loss += F.mse_loss(target_features[layer], original_features[layer])
                
                # compare it with style image
                style_loss = 0
                for layer in STYLE_LAYERS:
                    target_gram = gram_matrix(target_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += F.mse_loss(target_gram, style_gram)

                total_loss = CONTENT_COEFFICIENT * content_loss + STYLE_COEFFICIENT * style_loss

                # gradient descent
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step() 

                # clean up
                loss.append(total_loss.item())
                log.update()

        if DETAIL:
            with TaskLog("save loss to ./results/loss.png"):
                plt.plot(range(1, NUM_EPOCHS+1), loss)
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.savefig("./results/loss.png")
            with TaskLog("save processed style image to ./results/style.png"):
                save_processed_image(style_image, "./results/style.png")
            with TaskLog("save processed original image to ./results/original.png"):
                save_processed_image(original_image, "./results/original.png")
        with TaskLog(f"save target image to {SAVE_PATH}"):
            save_processed_image(target_image, SAVE_PATH)

if __name__ == "__main__":
    manager = Manager()
    manager.fetch_weights()
    manager.load_weights()
    manager.transfer()