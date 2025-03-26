import os 
import torch 
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vgg import vgg16 
from console import TaskLog, LoopLog
from image import preprocess_image, save_processed_image

device = "cuda" if torch.cuda.is_available() else "cpu"

def gram_matrix(feature):
    c, h, w = feature.size()
    features_reshaped = feature.view(c, h * w)
    gram = torch.mm(features_reshaped, features_reshaped.t())
    return gram.div(c * h * w)

class Manager:
    def __init__(self):
        with TaskLog("initialize Manager"):
            self.network = vgg16().to(device)

    def fetch_weights(self, save_dir):
        log = TaskLog("fetch vgg16 weights")
        log.start()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            torch_vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            torch.save(torch_vgg.state_dict(), f"{save_dir}/vgg16.pth")
            log.end()
        except Exception as e:
            log.error(e)
            exit(0)

    def load_weights(self, load_dir):
        log = TaskLog(f"load vgg16 weights")
        log.start()
        try:
            self.network.load_state_dict(torch.load(f"{load_dir}/vgg16.pth", weights_only=True))
            log.end()
        except Exception as e:
            log.error(e)
            exit(0)

    def transfer(self, style_path, target_path, save_path, num_epochs, detail=False):
        # define which layers to look at
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        content_layers = ['conv4_2']

        # preprocess style image and target(original) image
        with TaskLog(f"process style image {style_path}"):
            style_image = preprocess_image(style_path).to(device)
        with TaskLog(f"process target image {target_path}"):
            original_image = preprocess_image(target_path).to(device)

        # compute features of style image and original image
        with torch.no_grad():
            style_features = self.network.collect(style_image)
            style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}
            original_features = self.network.collect(original_image)


        target_image = original_image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target_image], lr=0.01)
        loss = []

        with LoopLog("transfer style", count=num_epochs, precise=True) as log:
            for epoch in range(num_epochs):
                # compute current features
                target_features = self.network.collect(target_image)

                # compare it with original image
                content_loss = 0
                for layer in content_layers:
                    content_loss += F.mse_loss(target_features[layer], original_features[layer])
                
                # compare it with style image
                style_loss = 0
                for layer in style_layers:
                    target_gram = gram_matrix(target_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += F.mse_loss(target_gram, style_gram)

                total_loss = 1 * content_loss + 1e6 * style_loss

                # gradient descent
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step() 

                # clean up
                loss.append(total_loss.item())
                log.update()

        if detail:
            with TaskLog("save loss to ./results/loss.png"):
                plt.plot(range(1, num_epochs+1), loss)
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.savefig("./results/loss.png")
            with TaskLog("save style image to ./results/style.png"):
                save_processed_image(style_image, "./results/style.png")
            with TaskLog("save original image to ./results/original.png"):
                save_processed_image(original_image, "./results/original.png")
        with TaskLog(f"save target image to {save_path}"):
            save_processed_image(target_image, save_path)

if __name__ == "__main__":
    manager = Manager()
    manager.fetch_weights(save_dir="./weights")
    manager.load_weights(load_dir="./weights")
    manager.transfer(style_path="./images/style/monet.jpg", target_path="./images/target/tree.jpg", save_path="./results/transfer.png", num_epochs=5000, detail=False)