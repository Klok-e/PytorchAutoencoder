import numpy as np
import matplotlib.pyplot as plt
import loader as ld
import torch.nn as nn
import torch.autograd
import torch
import torch.optim as optim
import random
import model as m


def main():
    images = ld.get_images_batch(0) + ld.get_images_batch(1) + ld.get_images_batch(2) + ld.get_images_batch(
        3) + ld.get_images_batch(4)
    ld.normalize(images)

    model = m.Autoencoder().cuda()
    model.load_state_dict(torch.load("autoencoder_save.pth"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # plots
    plt.ion()

    # prepare data
    inputs_batches = []
    for i in range(len(images)):
        input = torch.from_numpy(ld.resize_from_32x32x3_to_3x32x32(images[i])).cuda()
        input.unsqueeze_(0)
        inputs_batches.append(input)
    batch_size = 1000

    inputs_batches = [inputs_batches[i:i + batch_size] for i in range(0, len(inputs_batches), batch_size)]
    for i in range(len(inputs_batches)):
        inputs_batches[i] = torch.cat(inputs_batches[i], 0)

    num_batches = len(inputs_batches)
    for epoch in range(100):
        random.shuffle(inputs_batches)
        for i in range(len(inputs_batches)):
            # get the inputs
            batch = inputs_batches[i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            if i % (num_batches // 2) == 0:
                # print statistics
                print('epoch: %d, batch: %5d loss: %.3f' % (epoch, i, loss.item()))

                # plot
                rand_ind = random.randrange(0, batch_size)
                data_np1 = batch.cpu().data.numpy()[rand_ind]
                data_np2 = outputs.cpu().data.numpy()[rand_ind]

                inp_img = ld.resize_from_3x32x32_to_32x32x3(data_np1)
                outp_img = ld.resize_from_3x32x32_to_32x32x3(data_np2)
                st = np.concatenate([inp_img, outp_img], 1)
                plt.imshow(st)
                plt.pause(0.1)
            del loss
        torch.save(model.state_dict(), "my_autoencoder_" + str(epoch) + " epoch.pth")

    print('Finished Training')


if __name__ == "__main__":
    main()
