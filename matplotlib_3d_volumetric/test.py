# sudo apt-get install python3-matplotlib
# pip3 install scikit-image nibabel

import matplotlib.pyplot as plt
from skimage import data
import time

print('start astronaut')
astronaut = data.astronaut()
print('start chemistry')
ihc = data.immunohistochemistry()
print('start hubble')
hubble = data.hubble_deep_field()

# Initialize the subplot panels side by side
print('start subplot')
#fig, ax = plt.subplots(nrows=1, ncols=3)
#fig, ax = plt.subplots(nrows=7, ncols=9, gridspec_kw={'wspace':30, 'hspace':30},squeeze=True)
#fig, ax = plt.subplots(nrows=7, ncols=9, gridspec_kw={'wspace':30, 'hspace':30},squeeze=False)
fig, ax = plt.subplots(nrows=5, ncols=12)

# Show an image in each subplot
print('show plot')
#ax[0].imshow(astronaut)
#ax[0].set_title('Natural image')
#ax[1].imshow(ihc)
#ax[1].set_title('Microscopy image')
#ax[2].imshow(hubble)
#ax[2].set_title('Telescope image')

#print('sleep 20')
#time.sleep(20)
'''
Interlude: Getting The Data…
We’re going to download a dataset described in Buchel and Friston, Cortical Interactions Evaluated with Structural Equation Modelling and fMRI (1997). First, we create a temporary directory in which to download the data. We must remember to delete it when we are done with our analysis! If you want to keep this dataset for later use, change d to a more permanent directory location of your choice.
'''
import tempfile

# Create a temporary directory
d = tempfile.mkdtemp()

import os

# Return the tail of the path
#os.path.basename('http://google.com/attention.zip')

from urllib.request import urlretrieve

# Define URL
#url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip'

# Retrieve the data
#fn, info = urlretrieve(url, os.path.join(d, 'attention.zip'))

#import zipfile

# Extract the contents into the temporary directory we created earlier
#zipfile.ZipFile(fn).extractall(path=d)

# List first 10 files
#[f.filename for f in zipfile.ZipFile(fn).filelist[:10]]
#sys.exit(1)
'''
These are in the NIfTI file format, and we’ll need a reader for them. Thankfully, the excellent nibabel library provides such a reader. Make sure you install it with either conda install -c conda-forge nibabel or pip install nibabel, and then:
'''

import nibabel

'''
Now, we can finally read our image, and use the .get_data() method to get a NumPy array to view:
'''
# Read the image 
#struct = nibabel.load(os.path.join(d, 'attention/structural/nsM00587_0002.hdr'))
struct = nibabel.load('./attention/structural/nsM00587_0002.hdr')

# Get a plain NumPy array, without all the metadata
struct_arr = struct.get_data()

# Tip: if you want to directly continue to plotting the MRI data, execute the following lines of code:
#from skimage import io

#struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")

'''
… Back To Plotting
Let’s now look at a slice in that array:
'''
#plt.imshow(struct_arr[75])

'''
Whoa! That looks pretty squishy! That’s because the resolution along the vertical axis in many MRIs is not the same as along the horizontal axes. We can fix that by passing the aspect parameter to the imshow function:
'''
#plt.imshow(struct_arr[75], aspect=0.5)

'''
But, to make things easier, we will just transpose the data and only look at the horizontal slices, which don’t need such fiddling.
'''
struct_arr2 = struct_arr.T
#plt.imshow(struct_arr2[34])
print('len struct_arr2='+str(len(struct_arr2)))

for i in range(len(struct_arr2)):
  print('type for struct_array2['+str(i)+']='+str(type(struct_arr2[i])))
for i in range(5):
  for j in range(12):
    index=i*12+j
    ax[i,j].set_title(str(index).zfill(3))
    ax[i,j].imshow(struct_arr2[index])
    #plt.imshow(struct_arr2[index], aspect=0.7)
  #ax[i].set_title('image'+str(i).zfill(3))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.75)
#plt.show()
#time.sleep(60)

'''
Pretty! Of course, to then view another slice, or a slice along a different axis, we need another call to imshow:
'''
#plt.imshow(struct_arr2[5])

'''
All these calls get rather tedious rather quickly. For a long time, I would view 3D volumes using tools outside Python, such as ITK-SNAP. But, as it turns out, it’s quite easy to add 3D “scrolling” capabilities to the matplotlib viewer! This lets us explore 3D data within Python, minimizing the need to switch contexts between data exploration and data analysis.


The key is to use the matplotlib event handler API, which lets us define actions to perform on the plot — including changing the plot’s data! — in response to particular key presses or mouse button clicks.

In our case, let’s bind the J and K keys on the keyboard to “previous slice” and “next slice”:
'''
def previous_slice():
    pass

def next_slice():
    pass

def process_key(event):
    if event.key == 'j':
        previous_slice()
    elif event.key == 'k':
        next_slice()

'''
Simple enough! Of course, we need to figure out how to actually implement these actions and we need to tell the figure that it should use the process_key function to process keyboard presses! The latter is simple: we just need to use the figure canvas method mpl_connect:
'''
fig, ax = plt.subplots()
ax.imshow(struct_arr[..., 43])
fig.canvas.mpl_connect('key_press_event', process_key)

'''
You can find the full documentation for mpl_connect here, including what other kinds of events you can bind (such as mouse button clicks).

It took me just a bit of exploring to find out that imshow returns an AxesImage object, which lives “inside” the matplotlib Axes object where all the drawing takes place, in its .images attribute. And this object provides a convenient set_array method that swaps out the image data being displayed! So, all we need to do is:

plot an arbitrary index, and store that index, maybe as an additional runtime attribute on the Axes object.
provide functions next_slice and previous_slice that change the index and uses set_array to set the corresponding slice of the 3D volume.
use the figure canvas draw method to redraw the figure with the new data.
'''
def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

'''
try it
'''
multi_slice_viewer(struct_arr2)

'''
This works! Nice! But, if you try this out at home, you’ll notice that scrolling up with K also squishes the horizontal scale of the plot. Huh? (This only happens if your mouse is over the image.)


What’s happening is that adding event handlers to Matplotlib simply piles them on on top of each other. In this case, K is a built-in keyboard shortcut to change the x-axis to use a logarithmic scale. If we want to use K exclusively, we have to remove it from matplotlib’s default key maps. These live as lists in the plt.rcParams dictionary, which is matplotlib’s repository for default system-wide settings:
'''
'''
where pressing any of the keys in the list (i.e. <key1> or <key2>) will cause <command> to be executed.

Thus, we’ll need to write a helper function to remove keys that we want to use wherever they may appear in this dictionary. (This function doesn’t yet exist in matplotlib, but would probably be a welcome contribution!)
'''
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

# Now, we should be able to view all the slices in our MRI volume without pesky interference from the default keymap!
multi_slice_viewer(struct_arr2)

#plt.savefig('output.png')



plt.show()
time.sleep(60)

