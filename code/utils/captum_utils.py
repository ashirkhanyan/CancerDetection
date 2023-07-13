import matplotlib.pyplot as plt
import numpy as np
import torch


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


def plot_boxes(json_shape, outbox, image, actual, predicted, path, detection):
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF
    
    fig, ax = plt.subplots()
    # Convert the tensor to a NumPy array while preserving the shape
    image_array = TF.to_pil_image(image)
    image_array = TF.to_tensor(image_array)
    # Transpose the array to match the required shape 
    # for matplotlib (height, width, channels)
    image_array = image_array.numpy()
    img = image_array.transpose(1, 2, 0)
    ax.imshow(img)
    # add  the ground truth box
    x1, y1, x2, y2 = json_shape.numpy()
    width, height = x2 - x1, y2 - y1
    font_properties = {'size': 8, 'weight': 'bold'}
    
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 10, actual.replace("_", " "), color='r', fontdict = font_properties)
    # add  the model box
    x3, y3, x4, y4 = outbox.detach().numpy()
    width, height = x4 - x3, y4 - y3
    rect2 = patches.Rectangle((x3, y3), width, height, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect2)
    ax.text(x3, y3, predicted.replace("_", " "), color='g',  fontdict = font_properties)
    # add the model output box
    #ax.axis('off')
    #plt.suptitle(title)
    fig.savefig(path + '/' + detection + predicted + '.png')
    plt.close()

def get_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    # Calculate area of intersection rectangle
    intersection_area = torch.clamp(x2 - x1 , min=0) * torch.clamp(y2 - y1 , min=0)
    # Calculate areas of the two bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] )
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] )
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    # Calculate IoU
    iou = intersection_area / union_area
    return iou


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = x - (w / 2)
    y1 = y - (h / 2)
    x2 = x + (w / 2)
    y2 = y + (h / 2)

    bbox_modified = torch.tensor([x1, y1, x2, y2])

    return bbox_modified
