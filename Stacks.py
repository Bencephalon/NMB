#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:53:28 2017

@author: jungbt
"""
import os
import shutil
import math
import glob
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, color
from sklearn.mixture import GaussianMixture
from skimage.morphology import binary_dilation, remove_small_objects

class Slice():
    """
    Contains information pertinent to any kind of 2D data. Designed to be fed into a 
    Stack() object.
    """
    def __init__(self,directory,name):
        """
        Extracts name data from given file_name.
        File does not have to exist.
        """
        self.name = name
        self.ext = os.path.splitext(self.name)[1]
        self.root = os.path.splitext(self.name)[0]
        if self.ext == '.gz':
            self.ext = os.path.splitext(self.root)[1] + self.ext
            self.root = os.path.splitext(self.root)[0]
        self.dir = os.path.abspath(directory) + '/'
        self.path = self.dir + self.name
    def __str__(self):
        return (self.name)
    def rename(self,new_dir,new_name,remove=0):
        """Reinitialize Slice object with new location and name information"""
        self.__init__(new_dir,new_name)
class Stack():
    """
    A collection of Slice objects and associated data. Contains methods for 
    Slice manipulation and information retrevial.
    """
    def __init__(self,directory,pattern='*'):
        """
        Scans the provided directory for files matching the given pattern.
        Organizes all relevant files into Slice objects and contains information
        relevant to all associated Slices.
        """
        self.slices = []
        self.dir = os.path.abspath(directory) + '/'
        for img in glob.glob(self.dir + pattern):
            img = os.path.split(img)[1]
            self.slices.append(Slice(self.dir,img))
        self.slices = sorted(self.slices,key=lambda x: x.name)
        if len(self.slices) == 0:
            warnings.warn("Empty Volume.")
        elif not all([x for x in self.get_slice_attributes('ext') if x == self.get_slice_attributes('ext')[0]]):
            warnings.warn("Not all slices of Volume are of same type. Consider changing your pattern variable.")
    def __str__(self):
        return 'Volume object containing {} slices'.format(len(self.slices))
    def __repr__(self):
        return 'Volume object containing {} slices'.format(len(self.slices))
    def rename(self,new_dir,new_prefix=''):
        """Renames each Slice() in the Stack as [new_dir]/[new_prefix][current_name]"""
        for i in self.slices:
            i.rename(new_dir,new_prefix + i.name)
        self.dir = new_dir
    def exclude(self,img):
        """Removes file from Stack"""        
        if img in self.get_slice_attributes('name'):
            self.slices = [x for x in self.slices if x.name != img]
            #self.root = [os.path.splitext(x)[0] for x in self.slices]
            print(img + ' removed from Volume')
        else:
            print(img + ' does not exist in Volume')
        
    def include(self,img):
        """Adds file from Stack"""   
        if img not in self.get_slice_attributes('name'):
             self.slices = sorted(self.slices + Slice(self.dir,img),key=lambda x: x.name)
             print(img + ' added to Volume')
        else:
             print(img + ' already exists in Volume')
    def get_slice_attributes(self,attribute):
        """Returns a list of values for a given attribute from each Slice()"""
        return [getattr(x,attribute) for x in self.slices]
class NIFTI_Stack(Stack):
    """
    A collection of Slice objects and associated data. Contains methods for processing 
    NIFTI files and for converting a NIFTI_Stack into a Volume.
    Inherits Slice manipulation and information retrevial methods from Stack().
    """
    def __init__(self,directory,pattern='*.nii*',affine_3D=np.eye(4)):
        """
        Scans the provided directory for files matching the given pattern.
        Organizes all relevant files into Slice objects and contains information
        relevant to all associated Slices.
        """
        super().__init__(directory,pattern)
        self.affine_3D = affine_3D
        self.pix_dim = 3*[0]
        self.pix_dim[0] = sum(self.affine_3D[0,:])
        self.pix_dim[1] = sum(self.affine_3D[1,:])
        self.pix_dim[2] = sum(self.affine_3D[2,:])
    def volumize(self,out,overwrite=False):
        """
        Converts a series of 2D NIFTI files into a single 3D NIFTI file in RAS+ orientation.
        """
        if overwrite == False:
            #Stop if NIFTI volume exists
            if os.path.isfile(out):
                print(' - NIFTI Volume {} already exists'.format(out))
                return out
        #Load all Slice objects and append 2D arrays into 3D volume
        volume = nib.load(self.slices[0].path)
        volume = volume.get_data()
        volume = np.expand_dims(volume,2)
        for i in range(1,len(self.slices)-1):
            nii_img = nib.load(self.slices[i].path)
            nii_img = nii_img.get_data()
            nii_img = np.expand_dims(nii_img,2)
            volume = np.append(volume,nii_img,2)
        #Create NIFTI file
        volume = nib.Nifti1Image(volume, affine=self.affine_3D)
        nib.save(volume,out)
        print('Generated NIFTI Volume {}'.format(out))
        return out
class IMAGE_Stack(Stack):
    """
    A collection of Slice objects and associated data. Class contains method useful
    for preprocessing and calculating metadata for neuroanatomical images stored
    in common image formats (e.g. .png, .jpg).
    IMAGE_Stack also contains a method for converting data in an IMAGE_Stack to the
    NIFTI format and can output a NIFTI_Stack object with the same information.
    Inherits Slice manipulation and information retrevial methods from Stack().
    """
    def __init__(self,directory,pattern='*',orientation='XXX',pix_dim=[-1,-1,-1]):
        """
        Scans the provided directory for files matching the given pattern.
        Organizes all relevant files into Slice objects and contains information
        relevant to all associated Slices.
        """
        #Make a list of slices relevant to pattern criterea
        super().__init__(directory,pattern)
        #Add user defined metadata
        self.orientation = orientation
        self.pix_dim = pix_dim
        #Make note of pattern 
        self.pattern = pattern
        # Calculate affine matrix from user defined orientation and pix_dim
        self.affine_3D = self.calculate_affine_3D()
    def calculate_affine_3D(self):
        """
        Calculates an affine transformation necessary to transform the user-provided 
        orientation to the NIFTI standard RAS+ orientation.
        Returns an affine matrix.
        """
        #Generate empty affine
        aff = np.zeros((4,4))
        aff[3,3] = 1
        #Interpret user provided orientation string
        #Find the axis number for the Left-Right axis.
        x = self.orientation.find('R')
        if x == -1:
            x = self.orientation.find('L')
            aff[0,x] = -1*self.pix_dim[x]
        else:
            aff[0,x] = self.pix_dim[x]
        #Find the axis number for the Anterior-Posterior axis.
        y = self.orientation.find('A')
        if y == -1:
            y = self.orientation.find('P')
            aff[1,y] = -1*self.pix_dim[y]
        else:
            aff[1,y] = self.pix_dim[y]
        #Find the axis number for the Superior_Inferior axis.
        z = self.orientation.find('S')
        if z == -1:
            z = self.orientation.find('I')
            aff[2,z] = -1*self.pix_dim[z]
        else:
            aff[2,z] = self.pix_dim[z]
        return aff
    def convert_to_nifti(self,out_dir,prefix,overwrite=False):
        """
        Takes all images in the IMAGE_Stack and converts them to a NIFTI format
        with the name [prefix][original_image_name].nii.gz in the directory out_dir
        Returns a NIFTI_Stack object.
        """
        out_dir = os.path.abspath(out_dir) + '/'
        if overwrite == False:
            #Check if all relevant NIFTI files already exist.
            for i in range(len(self.slices)):
                if not os.path.isfile(out_dir + self.slices[i].root + '.nii.gz'):
                    break
            else:
                print(' - All NIFTI Files Exist. Utilizing currently existing data.')
                new_Stack = NIFTI_Stack(out_dir,prefix + '**.nii.gz',self.affine_3D)
                return new_Stack
        for i in self.slices:
                print('Converting {} Image to NIFTI Format'.format(i.name))
                #read image data
                img = cv2.imread(i.path,0)
                #Generate an affine matrix. The affine matrix contains orientation and dimension
                #information about the NIFTI file. 
                #NOTE: FOR 2D NIFTI FILES, the x and y axes must have a an axis length > 1
                #for registration tools to work. Therefore, this affine is not representative of
                #the final affine that will be used to set the orientation of our volume.
                #In the final volume: x = Left-Right axis, y= Anterior-Posterior axis, and 
                #z = Inferior-Superior axis. This is not necessarily true in this affine. For more information
                #on affines, please see the official NIFTI and Nibabel documentation
                aff_2D = np.eye(4)
                #voxel size of image axis 0
                aff_2D[0,0] = self.pix_dim[0]
                #voxel size of image axis 1
                aff_2D[1,1] = self.pix_dim[1]
                #voxel size of image axis 2 (distance between slices)
                aff_2D[2,2] = self.pix_dim[2]
                #Generate a NIFTI file using the image data array and the affine.
                img = nib.Nifti1Image(img, affine=aff_2D)
                nib.save(img,out_dir+prefix+i.root+'.nii.gz')
        return NIFTI_Stack(out_dir,prefix + '**.nii.gz',self.affine_3D)
    def zpad(self,out_dir,overwrite=False):
        """
        Pads edges of Slices in Stack to have equal lengths.
        Replaces existing Slice data in Stack to incorporate padding.
        Padding necessary for conversion of 2D slices into 3D volume.
        """
        out_dir = os.path.abspath(out_dir) + '/'
        #Keep track of minimum and maximum image axis sizes
        min_x = 999999999
        min_y = 999999999
        max_x = 0
        max_y = 0
        if overwrite == False:
            #Check if all relevant files exist.
            for i in range(len(self.slices)):
                if not os.path.isfile(out_dir + self.slices[i].name):
                    break
            else:
                print(' - All Padded Files Exist. Utilizing currently existing data.')
                #Alter Stack data to include padded files.
                for i in range(len(self.slices)):
                    self.slices[i].rename(out_dir,self.slices[i].name)    
                self.dir = out_dir
                return
        for i in range(len(self.slices)):
            #Read each image, tracking min and max axis size
            img = plt.imread(self.slices[i].path)
            if img.shape[0] > max_y:
                max_y = img.shape[0]
            if img.shape[1] > max_x:
                max_x = img.shape[1]
            if img.shape[0] < min_y:
                min_y = img.shape[0]
            if img.shape[1] < min_x:
                min_x = img.shape[1]
                
        if min_x == max_x and min_y == max_y:
            #All images are the same dimensions. Skipping padding.
            print('Images are all of same dimensions. No padding')
            for i in range(len(self.slices)):
                print('Moving {} to Output Directory'.format(self.slices[i].name))
                #Alter Stack data to include copied files.
                shutil.copyfile(self.slices[i].path,out_dir + self.slices[i].name)
                self.slices[i].rename(out_dir,self.slices[i].name)
        else:
            for i in range(len(self.slices)):
                print('Padding {}'.format(self.slices[i].name))
                #Open each image and pad with zeroes until axes are the same between images
                img = plt.imread(self.slices[i].path)
                pad_top = math.ceil((max_y - img.shape[0])/2)
                pad_bottom = math.floor((max_y - img.shape[0])/2)
                pad_left = math.ceil((max_x - img.shape[1])/2)
                pad_right = math.floor((max_x - img.shape[1])/2)
                padded_img = np.lib.pad(img, ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)), 'constant')
                padded_img = np.fliplr(padded_img) #####NOTE 12/20/2017: THIS IS A SEGMENT OF CODE TO DEAL WITH AN ERROR IN IMAGE CAPTURE. INCLUDE MORE ROBUST ORIENTATION FEATURES IN FUTURE VERSION.
                plt.imsave(out_dir + self.slices[i].name,padded_img[:,:,:3])
                #Alter Stack data to include padded files.
                self.slices[i].rename(out_dir,self.slices[i].name)    
        self.dir = out_dir
        return [max_x, max_y]
    def segmentation(self,out_dir,sampling_rate=0.01,clusters=3,overwrite=False):
        """
        Segmentation of RGB histological images using a Gaussian Mixed Model. 
        Slices inside stack will be masked by the relevant cluster and output into  out_dir
        Slices in Stack object will be replaced with their segmented versions.
        Lower sampling rate to reducing processing time.
        Defaults to a sampling rate of 1% and 3 clusters.
        """
        out_dir = os.path.abspath(out_dir) + '/'
        if overwrite == False:
            #Check if all relevant files exist.
            for i in range(len(self.slices)):
                if not os.path.isfile(out_dir + self.slices[i].name):
                    break
            else:
                print(' - All Segmented Files Exist. Utilizing currently existing data.')
                #Alter Stack data to include segmented files.
                for i in range(len(self.slices)):
                    self.slices[i].rename(out_dir,self.slices[i].name)    
                self.dir = out_dir
                return
        #Start Gaussian Mixture Models
        gauss = GaussianMixture(clusters)
        gauss_sample = np.zeros([1,3],'uint8')
        print('Sampling Image Data')
        #Convert subset of image data from RGB to a 1D LAB array
        for i in range(len(self.slices)):
            print(self.slices[i].name)
            img = io.imread(self.slices[i].path)
            img = img[:,:,:3]
            img_1D = img.reshape(img.shape[0]*img.shape[1],3)
            img_1D = img_1D[np.random.choice(list(range(len(img_1D))),size=int(len(img_1D)*sampling_rate))]
            lab_1D = color.rgb2lab(img_1D.reshape(img_1D.shape[0],1,img_1D.shape[1]))
            gauss_sample = np.r_[gauss_sample,lab_1D.reshape(img_1D.shape[0],3)]  
        print('Fitting Gaussian Mixed Model')
        #Fit model for all Slices in Stack.
        gauss.fit(gauss_sample)
        print('Applying Gaussian Mixed Model')
        #Open images and mask by a specific cluster. 
        ##NOTE 12/23/2017: 
        ###NEED TO INCLUDE SUPPORT FOR MULTIPLE CLUSTERS AT A LATER DATE.
        ###CLUSTER SELECTION REQUIRES USER INPUT. THIS WILL NOT WORK ON BIOWULF.
        ###WORKAROUND: RUN SEGMENTATION INTERACTIVELY, THEN RUN REST OF PIPELINE 
            ###ON SBATCH WITH pipe.input.overwrite = False
        for i in range(len(self.slices)):
            #open and modify image data for GMM fit
            img = io.imread(self.slices[i].path)
            img = img[:,:,:3]
            lab = color.rgb2lab(img[:,:,:3])
            img_1D = lab.reshape(lab.shape[0]*lab.shape[1],3)
            #Generate mask from clusters
            mask_1D = gauss.predict(img_1D)
            mask = mask_1D.reshape(lab.shape[0],lab.shape[1])
            #First instance: User input to identify cluster containing histology
            if i == 0:
                for m in range(clusters):
                    #Generate boolean mask for each cluster.
                    mask_binary = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
                    mask_binary[mask != m] = False
                    mask_binary[mask == m] = True
                    mask_binary = mask_binary.reshape(mask.shape)
                    mask_binary_d = remove_small_objects(mask_binary.astype(bool),min_size=64)
                    mask_binary_d = binary_dilation(mask_binary_d).astype(img.dtype)
                    #Generate masked images from each cluster.
                    print('Cluster {}'.format(m))
                    masked_rgb = np.array(np.zeros(img.shape))
                    for j in range(clusters):
                        masked_rgb[:,:,j] = mask_binary_d*img[:,:,j]
                    plt.imshow(masked_rgb.astype(img.dtype),cmap=plt.cm.binary)
                    plt.show()
                #Prompt user to select cluster.
                k = input("Which cluster contains histology?")
                k = int(k)
            #Generate boolean mask for selected cluster.
            mask_binary = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
            mask_binary[mask != k] = False
            mask_binary[mask == k] = True
            mask_binary = mask_binary.reshape(mask.shape)
            mask_binary_d = remove_small_objects(mask_binary.astype(bool),min_size=64)
            mask_binary_d = binary_dilation(mask_binary_d).astype(img.dtype)
            #Mask image by boolean mask
            masked_rgb = np.array(np.zeros(img.shape))
            for j in range(3):
                masked_rgb[:,:,j] = mask_binary_d*img[:,:,j]
            plt.imsave(out_dir + self.slices[i].name,masked_rgb.astype(img.dtype))
            #Modify Stack data to incorporate new segmented data.
            self.slices[i].rename(out_dir,self.slices[i].name)
        self.dir = out_dir