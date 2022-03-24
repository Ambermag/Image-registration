# Image-registration

!pip install simpleitk
!pip install matplotlib

import numpy as np
import SimpleITK as sitk
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import clear_output


fixed_image = sitk.ReadImage("MRI.mha")
moving_image = sitk.ReadImage("MRI.mha")

# Get Spacing: How large each voxal in mm, each voxal is xx mm wide along x, y and z axis. --> so it is not isotrophic because the spacing are not same.  
# Size: pixels along the directions
# Origing: Where the image located with respect to the imaginary global coordinate frame. 
# Direction: How the image located on local coordinate frame.


print("Image spacing is:", fixed_image.GetSpacing())
print("Image size is:", fixed_image.GetSize())
print("Image origin is:", fixed_image.GetOrigin())
print("Image direction is:", fixed_image.GetDirection())

# The loaded SITK image can now be simply converted to an nd numpy array: the array order is reversed. 
# Also the channels will be, as if your images flipped all the directions. 
# Note: The channel order is reversed when converting the image to a numpy array

fixed_image_np = sitk.GetArrayFromImage(fixed_image)
print("The numpy array image is now of fixed image:", fixed_image_np.shape)

# Now that we have access to the numpy translated image, we can extract a slice out of the image. let's visualize a slice of this MRI image for slice 100 on the x direction and take the other directions empty. 
# DO: Check the Y axis for the rotation example. 


fixed_image_np = sitk.GetArrayFromImage(fixed_image)
moving_image_np = sitk.GetArrayFromImage(moving_image)
slice_fixed = fixed_image_np[100, :, :]
slice_moving = moving_image_np[100, :, :]


plt.rcParams['figure.figsize'] = [15, 7]

plt.imshow(np.hstack((slice_fixed, slice_moving)))
plt.show()
plt.imshow(slice_fixed)
plt.imshow(slice_moving, cmap='gray', alpha=0.5)

# We can now apply an affine transformation on the input MRI but first let's define a function to calculate a rotation matrix based on input Euler angles. 
# Rotation Matrix based on 3 input elements, rotation around x,y,z axis. Based on the 3 matrixes, multiplication of these 3 returns rotation matrix. 

def getRotationMatrix(rx=0, ry=0, rz=0):
    rx *= np.pi / 180
    ry *= np.pi / 180
    rz *= np.pi / 180

    matx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    maty = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    matz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    return matx @ maty @ matz 
    
# By calling this function and applying: rotation 10degrees around x, -20degrees around y and 15 degrees around z - we get a rotation matrix stored.
Then we create a transformed object, using simpleitk. We use affine (define dimentionality as:3) : 3d

print(getRotationMatrix)

rotation_matrix = getRotationMatrix(rx=10, ry=-20, rz=15)

transform = sitk.AffineTransform(3)
print("The transformation will have", transform.GetNumberOfParameters(), "Parameters.")

transform.SetMatrix(rotation_matrix.ravel())
transform.SetTranslation([10, 50, -14])[Image registration Brainmha.zip](https://github.com/Ambermag/Image-registration/files/8343992/Image.registration.Brainmha.zip)


initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
print(initial_transform)
