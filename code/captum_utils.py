import matplotlib.pyplot as plt
import numpy as np


def visualize_attr_maps(path, X, y, class_names, attributions, titles,
                        attr_preprocess=lambda attr: attr.permute(1, 2, 0).detach().numpy(),
                        cmap='viridis', alpha=0.7):
    """
    A helper function to visualize captum attributions for a list of captum attribution algorithms.
    path (str): name of the final saved image with extension (note: if batch of images are in X, 
                      all images/plots saved together in one final output image with filename equal to path)
    X (numpy array): shape (N, H, W, C)
    y (numpy array): shape (N,)
    class_names (dict): length equal to number of classes
    attributions(A list of torch tensors): Each element in the attributions list corresponds to an
                      attribution algorithm, such as Saliency, Integrated Gradient, Perturbation, etc.
    titles(A list of strings): A list of strings, names of the attribution algorithms corresponding to each element in
                      the `attributions` list. len(attributions) == len(titles)
    attr_preprocess: A preprocess function to be applied on each image attribution before visualizing it with
                      matplotlib. Note that if there are a batch of images and multiple attributions 
                      are visualized at once, this would be applied on each individual image for each attribution
                      i.e. attr_preprocess(attributions[j][i])
    """

    N = attributions[0].shape[0]
    plt.figure()
    for i in range(N):
        plt.subplot(len(attributions) + 1, N + 1, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i].item()])

    plt.subplot(len(attributions) + 1, N + 1, N + 1)
    plt.text(0.0, 0.5, 'Original Image', fontsize=14)
    plt.axis('off')
    for j in range(len(attributions)):
        for i in range(N):
            plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + i + 1)
            attr = np.array(attr_preprocess(attributions[j][i]))
            attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
            attr = attr * 0.2 + 0.5
            attr = attr.clip(0.0, 1.0)
            plt.imshow(attr, cmap=cmap, alpha=alpha)
            plt.axis('off')
        plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + N + 1)
        plt.text(0.0, 0.5, titles[j], fontsize=14)
        plt.axis('off')

    plt.gcf().set_size_inches(20, 13)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def compute_attributions(algo, inputs, **kwargs):
    """
    A common function for computing captum attributions
    """
    return algo.attribute(inputs, **kwargs)


def plot_boxes(self, json_shape, outbox, images, path, order, batch_size, ngraphs = 2):
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF
    im_idx = np.random.choice(batch_size, size=ngraphs, replace=False)
    for i in range(ngraphs):
        fig, ax = plt.subplots()
        # Convert the tensor to a NumPy array while preserving the shape
        image_array = TF.to_pil_image(images[im_idx[i]])
        image_array = TF.to_tensor(image_array)
        # Transpose the array to match the required shape 
        # for matplotlib (height, width, channels)
        image_array = image_array.numpy()
        img = image_array.transpose(1, 2, 0)
        ax.imshow(img)
        # add  the ground truth box
        x1, y1, x2, y2 = json_shape[im_idx[i]].numpy()
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        #ax[row, col].text(x1, y1 - 10, "Ground Truth", color='r')
        # add  the model box
        x3, y3, x4, y4 = outbox[im_idx[i]].detach().numpy()
        width, height = x4 - x3, y4 - y3
        rect2 = patches.Rectangle((x3, y3), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect2)
        #ax[row, col].text(x3, y3 + 10, "Model Prediction", color='g')
        # add the model output box
        #ax.axis('off')
        #plt.suptitle(title)
        fig.savefig(path + '/' + str(order) + str(i) + '.png')
        plt.close()


