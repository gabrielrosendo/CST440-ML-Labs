# https://datasets.activeloop.ai/docs/ml/datasets/lfw-deep-funneled-dataset/
import deeplake

ds = deeplake.load('hub://activeloop/lfw-deep-funneled')
print(ds)  # See dataset structure
print(ds.tensors)  # Get the number of images
print(ds.name[:10])  # Print the first 10 names

import matplotlib.pyplot as plt

image = ds.images[0].numpy()  # Convert first image to NumPy array
name = ds.name[0].data()  # Get corresponding name

plt.imshow(image)
plt.axis("off")
plt.title(name)
plt.show()
