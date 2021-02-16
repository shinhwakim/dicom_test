import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
#init_notebook_mode(connected=True) 
import random

'''
Then, let's specify a specific DICOM study we can take a closer look. Let's take a look at a chest CT stack from Kaggle which contains a lung cancer.

The whole dataset is 140GB unzipped, but each examination is only 70MB or so.

Here we'll use the patient ID 5267ea7baf6332f29163064aecf6e443 from that dataset, which has been labeled as positive for lung cancer.
'''
#data_path = "/data/LungCancer-data/stage1/train/cancer/5267ea7baf6332f29163064aecf6e443/"
#data_path = "./5267ea7baf6332f29163064aecf6e443/"
#data_path = "./temp/721949894f5309ed4975a67419230a3c/"
#data_path = "./temp/70671fa94231eb377e8ac7cba4650dfb/"
#data_path = "./temp/7027c0b8c8f8dcc76c6e4ba923d60a2e/"
#data_path = "./temp/71665cc6a7ee85268ca1da69c94bbaeb/"
#data_path = "./temp/7050f8141e92fa42fd9c471a8b2f50ce/"
#data_path = "./temp/713d8136c360ad0f37d6e53b61a7891b/"
data_path = "./temp/700bdc2723c2ac75a8a8376d9ca170ad/"
#data_path = "./temp/7180c83eb184d5c9dfcbda228ab91213/"
#data_path = "./temp/70f4eb8201e3155cc3e399f0ff09c5ef/"
#data_path = "./temp/7051fc0fcf2344a2967d9a1a5478208e/"
#data_path = "./temp/71e09cd11d743964f1abf442c34f2c9d/"
#data_path = "./temp/70287a7720e0d90249ac7b3978b7ca40/"
#data_path = "./temp/7191c236cfcfc68cd21143e3a0faac51/"
#data_path = "./temp/718f43ecf121c79899caba1e528bd43e/"
#data_path = "./temp/722429bc9cb25d6f4b7a820c14bf2ab1/"

#data_path = "./temp/7c2fd0d32df5a2780b4b10fdf2f2cdbe/"
#data_path = "./temp/7c8aa548b813dadf972c38e806320179/"
#data_path = "./temp/7ce310b8431ace09a91ededcc03f7361/"
#data_path = "./temp/7cf1a65bb0f89323668034244a59e725/"
#data_path = "./temp/7d46ce019d79d13ee9ce8f18e010e71a/"
#data_path = "./temp/7daeb8ef7307849c715f7f6f3e2dd88e/"
#data_path = "./temp/7dbc5207b9ec1a1921cc2f03f9b37684/"
#data_path = "./temp/7dc59759253943ac070ab2557c219731/"
#data_path = "./temp/7df28e2253be3490208ba9a9f470ea1d/"
#data_path = "./temp/7eb217c0444e5d866bd462ade5266a06/"
#data_path = "./temp/7ec258e536a1e0353375295ad1b71e5b/"
#data_path = "./temp/7f096cdfbc2fe03ec7f779278416a78c/"
#data_path = "./temp/7f137d30638a87d151ac7e84eeaf48e8/"
#data_path = "./temp/7f45518a2f938a92fa99658d98770316/"
#data_path = "./temp/7f524231183ed193b8f2e3d9cc73c059/"
#data_path = "./temp/7faa456389e1ffde464819d0b1360188/"
#data_path = "./temp/7fb1c8ffd78ca4b6869044251add36b4/"
#data_path = "./temp/7fd5be8ec9c236c314f801384bd89c0c/"
#data_path = "./temp/7ffe144af38f85d177af286c5d941ef1/"



#data_path = "./7c02c641324c598cd935b588189c87db/"

output_path = working_path = "/home/shin/Documents/"
g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))

'''
Here we make two helper functions.

load_scan will load all DICOM images from a folder into a list for manipulation.
The voxel values in the images are raw. get_pixels_hu converts raw values into Houndsfeld units
The transformation is linear. Therefore, so long as you have a slope and an intercept, you can rescale a voxel value to HU.
Both the rescale intercept and rescale slope are stored in the DICOM header at the time of image acquisition (these values are scanner-dependent, so you will need external information).
'''
#      
# Loop over the image files and store everything into a list.
# 

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

'''
This is a good time to save the new data set to disk so we don't have to reprocess the stack every time.
'''
np.save(output_path + "fullimages_%d.npy" % (id), imgs)

'''
Displaying Images
The first thing we should do is to check to see whether the Houndsfeld Units are properly scaled and represented.

HU's are useful because it is standardized across all CT scans regardless of the absolute number of photons the scanner detector captured. If you need a refresher, here's a quick list of a few useful ones, sourced from Wikipedia.

Substance	HU
Air	−1000
Lung	−500
Fat	−100 to −50
Water	0
Blood	+30 to +70
Muscle	+10 to +40
Liver	+40 to +60
Bone	+700 (cancellous bone) to +3000 (cortical bone)
Let's now create a histogram of all the voxel data in the study.
'''
file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

'''
Critiquing the Histogram
The histogram suggests the following:

There is lots of air
There is some lung
There's an abundance of soft tissue, mostly muscle, liver, etc, but there's also some fat.
There is only a small bit of bone (seen as a tiny sliver of height between 700-3000)
This observation means that we will need to do significant preprocessing if we want to process lesions in the lung tissue because only a tiny bit of the voxels represent lung.

More interestingly, what's the deal with that bar at -2000? Air really only goes to -1000, so there must be some sort of artifact.

Let's take a look at the actual images.

Displaying an Image Stack
We don't have a lot of screen real estate, so we'll be skipping every 3 slices to get a representative look at the study.
'''
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    show_every=min(show_every, int(len(stack)/rows/cols))
    print('show_every='+str(show_every))
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        print('ind='+str(ind)+' len(stack)='+str(len(stack)))
        if ind==len(stack):
          ind=len(stack)-1
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)

'''
So as it turns out, what we were seeing as HU=-2000 are the voxels outside of the bore of the CT. "Air," in comparison, appears gray because it has a much higher value. As a result, the lungs and soft tissue have somewhat reduced contrast resolution as well.

We will try to manage this problem when we normalize the data and create segmentation masks.

(By the way, did you see the cancer? It's on slices 97-112.)

Resampling
Although we have each individual slices, it is not immediately clear how thick each slice is.

Fortunately, this is in the DICOM header.
'''
print("Slice Thickness: %f" % patient[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))

'''
This means we have 2.5 mm slices, and each voxel represents 0.7 mm.

Because a CT slice is typically reconstructed at 512 x 512 voxels, each slice represents approximately 370 mm of data in length and width.

Using the metadata from the DICOM we can figure out the size of each voxel as the slice thickness. In order to display the CT in 3D isometric form (which we will do below), and also to compare between different scans, it would be useful to ensure that each slice is resampled in 1x1x1 mm pixels and slices.
'''
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)

'''
3D Plotting
Having isotropic data is helpful because it gives us a sense of the Z-dimension. This means we now have enough information to plot the DICOM image in 3D space. For kicks we'll focus on rendering just the bones.

Visualization Toolkit (VTK) is excellent for 3D visualization because it can utilize GPU for fast rendering. However, I can't get VTK to work in Jupyter, so we will take a slightly different approach:

Create a high-quality static using 3D capability of matplotlib
Create a lower-quality but interactive render using plotly, which has WebGL support via JavaScript.
The marching cubes algorithm is used to generate a 3D mesh from the dataset. The plotly model will utilize a higher step_size with lower voxel threshold to avoid overwhelming the web browser.
'''
def make_mesh(image, threshold=-300, step_size=1):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    #iplot(fig)

def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    #ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()

v, f = make_mesh(imgs_after_resamp, 350)
plt_3d(v, f)

##v, f = make_mesh(imgs_after_resamp, 350, 2)
##plotly_3d(v, f)

'''
Segmentation
If you are interested in chest CTs because you're interested in picking up lung cancers, you're not alone.

Machine learning algorithms work a lot better when you can narrowly define what it is looking at. One way to do this is by creating different models for different parts of a chest CT. For instance, a convolutional network for lungs would perform better than a general-purpose network for the whole chest.

Therefore, it is often useful to pre-process the image data by auto-detecting the boundaries surrounding a volume of interest.

The below code will:

Standardize the pixel value by subtracting the mean and dividing by the standard deviation
Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
Using Erosion) and Dilation) which has the net effect of removing tiny features like pulmonary vessels or noise
Identify each distinct region as separate image labels (think the magic wand in Photoshop)
Using bounding boxes for each image label to identify which ones represent lung and which ones represent "every thing else"
Create the masks for lung fields.
Apply mask onto the original image to erase voxels outside of the lung fields.
'''
#Standardize the pixel values
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
        #print('prop label='+str(prop.label)+' bbox='+str(B))
    #print('good_labels='+str(good_labels))
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img
'''
Single Slice Example At Each Step
We want to make sure the algorithm doesn't accidentally exclude cancer from the region of interest (due to its "soft tissue" nature). So let's test this out on a single slice.
'''
#img = imgs_after_resamp[260]
#img = imgs_after_resamp[-1]
for i in range(0, 10):
  index=random.randint(0, len(imgs_after_resamp)-1)
  img = imgs_after_resamp[index]
  make_lungmask(img, display=True)

  '''
  A Few Observations
  Compare the difference in contrast between the finished slice alongside the original. Not only is extrapulmonary data properly cleaned up, the contrast is also improved.
  
  If we were to apply a machine learning algorithm to the image stack, the algorithm would have a much easier time to identify a primary lung lesion. The Kaggle lung cancer data contains labeled cancer and no-cancer datasets that can be used for this training (and a $1MM bounty).
  
  Downsides of using this mask appropach is you can miss hilar/perihilar disease fairly easily.
  
  Apply Masks to All Slices
  The single-slice example seemed to work pretty well.
  
  Let's now apply the mask to all the slices in this CT and show a few examples.
  '''
  masked_lung = []
  
  for img in imgs_after_resamp:
      masked_lung.append(make_lungmask(img))
  
  sample_stack(masked_lung, show_every=10)
'''
Looks like things check out.

The lung lesion is properly preserved in the ROI, and it appears to work wel from lung bases all the way to the apices.

This would be a good time to save the processed data.
'''
np.save(output_path + "maskedimages_%d.npy" % (id), imgs)



