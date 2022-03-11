# Some modules in the code subfolder use 'Agg', 
#       which is not compatible with plotting 
#       grayscale images. This import snippet 
#       ensures grayscale plotting is possible.
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Grayscale plotting
plt.imshow(array, cmap='gray', vmin=0, vmax=255)
plt.show()

# Grayscale plotting with my image_transforms variables
for iV in range(len(values)):
        plt.imshow(values[iV], cmap='gray', vmin=0, vmax=255)
        plt.show()

# Plot the sparse vector field of the LocalDeform class
plt.quiver(du, dv)
plt.axis('off')
plt.show()
plt.clf()

# LocalDeform plotting for image and annotation
plt.imshow(np.hstack( (np.squeeze(image), 
                        np.squeeze(deformed_image), 
                        np.squeeze(image-deformed_image)
                        ), 
                        ), cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.show()

plt.imshow(np.hstack( (np.squeeze(annot), 
                        np.squeeze(deformed_annot), 
                        np.squeeze(annot-deformed_annot)
                        ), 
                        ), cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.show()
