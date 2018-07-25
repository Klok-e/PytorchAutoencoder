import model
import torch
import loader as ld
import numpy as np


def main():
    m = model.Autoencoder()
    m.load_state_dict(torch.load("autoencoder_save.pth"))

    decoder = m.decoder.eval()
    for i in range(100):  # 100 random samples
        rnd_latent_vector = torch.rand(1, 64, 1, 1) * 2 - 1
        output = decoder(rnd_latent_vector)
        image = ld.resize_from_3x32x32_to_32x32x3(np.squeeze(output.data.numpy(), 0))
        image = (image * 255).astype(np.uint8)
        ld.save_img(image, "generated data/cifar10_generated_image" + str(i))


if __name__ == "__main__":
    main()
