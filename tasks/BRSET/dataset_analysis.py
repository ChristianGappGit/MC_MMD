"""
BRSET dataset analysis
image: print an shape disztibution of the 2D images (jpg)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup paths
datadir = "/home/christian/data/BRSET"  #Server
#datadir = "/media/christian/Daten4/christian/PhD/PhD_BigData/Projekte/BRSET" #Local
img_path = f"{datadir}/images"

outdir = "./dataAnalysis"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

#Load data
#List images, if they end with .jpg
img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

#Get image shapes
img_shapes = []
for img_file in img_files:
    img = plt.imread(f"{img_path}/{img_file}")
    img_shapes.append(img.shape)

#Convert to numpy array
img_shapes = np.array(img_shapes)

#Print some statistics
print(f"Number of images: {img_shapes.shape[0]}")
print(f"Mean height: {np.mean(img_shapes[:,0])}")
print(f"Mean width: {np.mean(img_shapes[:,1])}")
print(f"Median height: {np.median(img_shapes[:,0])}")
print(f"Median width: {np.median(img_shapes[:,1])}")

#Plot histogram
plt.figure()
plt.hist(img_shapes[:,0], bins=50, alpha=0.5, label='height')
plt.hist(img_shapes[:,1], bins=50, alpha=0.5, label='width')
plt.legend()
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.title('Image size distribution')
plt.savefig(f"{outdir}/img_size_distribution.png")

#Plot scatter
plt.figure()
plt.scatter(img_shapes[:,0], img_shapes[:,1])
plt.xlabel('Height')
plt.ylabel('Width')
plt.title('Image size scatter')
plt.savefig(f"{outdir}/img_size_scatter.png")

#Plot boxplot
plt.figure()
plt.boxplot(img_shapes)
plt.xticks([1,2], ['Height', 'Width'])
plt.ylabel('Size')
plt.title('Image size boxplot')
plt.savefig(f"{outdir}/img_size_boxplot.png")

#END
print("Done")