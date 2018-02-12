#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:20:10 2017

@author: jungbt
"""
import numpy as np
from nipype.interfaces import afni
import nibabel as nib
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from nipype.interfaces.ants import Registration, N4BiasFieldCorrection,ApplyTransforms
from multiprocessing.dummy import Pool 
import warnings
import Stacks
import copy
import InputHelper
from skimage import color
##############################CLASSES##############################
class Pipeline():
    def __init__(self):
        """
        Initiates pipeline to accept input variables. 
        Input variables will be stored under Pipeline.input in the InputSpec Class.
        Define input variables by setting attributes under Pipeline.input.
        For help with input variables, run Pipeline.input.print_help()
        """
        self.input = InputHelper.InputSpec()
        self.input.print_help()
    def __str__(self):
        """ return documentation for all potential input variable """
        return self.input.print_help()
    def __repr__(self):
        """ return documentation for all potential input variable """
        return self.input.print_help()
    def run(self):
        """
        Use Pipeline.run() to check for requisite input variables. If all requirements
        are satisfied, pipeline will define variables and start processing data
        """
        if self.input.check_inputs() == 1:
            user_var = self.input.initialize_inputs()
            self.bf = user_var[0]
            self.histology = user_var[1]
            self.MRI = user_var[2]
            self.root_dir = user_var[3]
            self.overwrite = user_var[4]
            self.threads = user_var[5]
            self.reg_method = user_var[6]
            self.resolution_level = user_var[7]
            self.color = user_var[8]
            self.start_pipeline()
        else:
            self.input.print_inputs()
    def start_pipeline(self):
        """
        Defines the steps and order of the pipeline. Only call from Pipeline.run()
        to avoid errors.
        """
        mark = '================================='
        print(mark, '\nSTEP 0: GENERATING OUTPUT DIRECTORIES\n', mark)
        self.gen_directory_struct()
        print(mark, 'STEP 1:PREPROCESSING', mark)
        self.orig_MRI = self.MRI
        self.preprocess_histology()
        self.preprocess_blockface()
        self.preprocess_MRI()
        print(mark, '\nSTEP 2:RESAMPLING (OPTIONAL)\n', mark)
        #Resample data
        if self.resolution_level is not 'MRI':
            self.MRI_path += 'resampled_'
        if self.resolution_level == 'histology':
            skip_flag = False
            #Match the resolution of histology
            if self.overwrite == False and os.path.isfile(self.MRI_path + os.path.split(self.orig_MRI)[1]):
                print(' - {} Already Exists. Utilizing currently existing data.'.format(self.MRI_path + os.path.split(self.orig_MRI)[1]))
                self.MRI = self.MRI_path + os.path.split(self.orig_MRI)[1]
            else:
                self.resample(self.MRI,self.MRI_path + os.path.split(self.orig_MRI)[1],self.histology.pix_dim,self.histology.affine_3D,3)
                self.MRI = self.MRI_path + os.path.split(self.orig_MRI)[1]
            if self.overwrite == False:
                for slice in self.BF_NIFTI.slices:
                    if not os.path.isfile(self.orig_bf_loc + "/NIFTI/resampled/{}".format(slice.name)):
                        break
                else:
                    print(' - All Resampled Blockface NIFTI Files Exist. Utilizing currently existing data.')
                    self.BF_NIFTI.rename(self.orig_bf_loc + "/NIFTI/resampled/")
                    self.BF_NIFTI.affine_3D = self.hist_NIFTI.affine_3D
                    skip_flag = True
            if skip_flag == False:
                for slice in self.BF_NIFTI.slices:
                    self.resample(slice.path,self.orig_bf_loc + "/NIFTI/resampled/{}".format(slice.name),self.histology.pix_dim,self.histology.affine_3D,2)
                self.BF_NIFTI.rename(self.orig_bf_loc + "/NIFTI/resampled/")
                self.BF_NIFTI.affine_3D = self.hist_NIFTI.affine_3D
        elif self.resolution_level == 'blockface':
            #Match the resolution of ng
            if self.overwrite == False and os.path.isfile(self.MRI_path + os.path.split(self.orig_MRI)[1]):
                print(' - {} Already Exists. Utilizing currently existing data.'.format(self.MRI_path + os.path.split(self.orig_MRI)[1]))
            else:
                self.resample(self.MRI,self.MRI_path + os.path.split(self.orig_MRI)[1],self.bf.pix_dim,self.histology.affine_3D,3)
                self.MRI = self.MRI_path + os.path.split(self.orig_MRI)[1]
        print(mark, '\nSTEP 3:ALIGNMENT\n', mark)
        self.slice_by_slice_alignment(self.threads,self.orig_slice_by_slice_loc)
        self.blockface_to_MRI_alignment(self.orig_bf_loc + "/volume/aligned_to_MRI")
        self.orig_hist_NIFTI.col = 'gray'
        pool = Pool(processes=self.threads)
        pool.map(Transform_Wrapper(self.orig_hist_NIFTI,self.hist_transform,self.BF_NIFTI,self.orig_slice_by_slice_loc), list(range(len(self.hist_transform.slices))))
        pool.close()
        pool.join()
        hist_vol = Stacks.NIFTI_Stack(self.orig_slice_by_slice_loc + '/gray', '*.nii.gz',self.hist_NIFTI.affine_3D)
        hist_vol.volumize(self.final_out +'/hist_to_bf.nii.gz')
        self.final_apply_transform(self.final_out +'/hist_to_bf.nii.gz',self.final_out +'/hist_to_MRI.nii.gz')
        if self.color == True:
            print(mark, '\nSTEP 4:COLORIZATION\n', mark)
            self.colorize(self.orig_col_split_loc,self.final_out)
            
        print('Done!')
    def gen_directory_struct(self):
        """
        Generates all directories necessary for pipeline to store data in root_dir
        Defines commonly used paths for use by other methods in the pipeline
        """
        #Name commonly used paths
        orig_hist_loc = "Histology"
        orig_col_split_loc = orig_hist_loc +"/color_split"
        orig_bf_loc = "Blockface"
        orig_slice_by_slice_loc = orig_hist_loc + "/slice_by_slice_alignment"
        final_out = "output"
        #Populate names of all paths using commonly used paths
        dir_structure = [orig_hist_loc,
         orig_hist_loc +"/orig",
         orig_hist_loc +"/orig/orig",
         orig_hist_loc +"/orig/NIFTI",
         orig_hist_loc +"/color_split",
         orig_col_split_loc + "/Blue",
         orig_col_split_loc + "/Blue/NIFTI",
         orig_col_split_loc + "/Blue/orig",
         orig_col_split_loc + "/Green",
         orig_col_split_loc + "/Green/NIFTI",
         orig_col_split_loc + "/Green/orig",
         orig_col_split_loc + "/Red",
         orig_col_split_loc + "/Red/NIFTI",
         orig_col_split_loc + "/Red/orig",
         orig_hist_loc +"/segmented",
         orig_hist_loc +"/segmented/orig",
         orig_hist_loc +"/segmented/padded",
         orig_hist_loc +"/segmented/padded/orig",
         orig_hist_loc +"/segmented/padded/NIFTI",
         orig_slice_by_slice_loc,
         orig_slice_by_slice_loc +"/composite_transform",
         orig_slice_by_slice_loc +"/grayscale",
         orig_slice_by_slice_loc +"/gray",
         orig_slice_by_slice_loc +"/color",
         orig_slice_by_slice_loc +"/color/Blue",
         orig_slice_by_slice_loc +"/color/Green",
         orig_slice_by_slice_loc +"/color/Red",
         orig_slice_by_slice_loc +"/color/volumes",
         orig_bf_loc,
         orig_bf_loc + "/NIFTI",
         orig_bf_loc + "/NIFTI/orig",
         orig_bf_loc + "/NIFTI/resampled",
         orig_bf_loc + "/orig",
         orig_bf_loc + "/volume",
         orig_bf_loc + "/volume/aligned_to_MRI",
         orig_bf_loc + "/volume/orig",
         final_out,
         "MRI",]
        #Create directories if they don't exist
        try:
            os.makedirs(self.root_dir)
            print(" - Created {}".format(self.root_dir))
        except:
            print(' - {} Already Exists'.format(self.root_dir))
        for i in dir_structure:
            try:
                os.makedirs(self.root_dir + i)
                print(" - Created {}{}".format(self.root_dir,i))
            except:
                print(' - {}{} Already Exists'.format(self.root_dir,i))    
        #Define commonly used paths for use by other methods.
        self.orig_hist_loc = self.root_dir + orig_hist_loc
        self.orig_col_split_loc = self.root_dir + orig_col_split_loc
        self.orig_bf_loc = self.root_dir + orig_bf_loc
        self.orig_slice_by_slice_loc = self.root_dir + orig_slice_by_slice_loc
        self.final_out = self.root_dir + final_out
    def preprocess_histology(self):
        """
        Modifies histology slices to allow for alignment to blockface images.
        Converts histology images to NIFTI format, Splits RGB channels, 
        and converts histology to grayscale NIFTI format.
        """
        print('Preprocessing Histology')
        self.histology_color = copy.deepcopy(self.histology) #Make a copy of self.histology for separate color splitting
        self.histology_color.zpad(self.orig_hist_loc + '/orig/orig/',self.overwrite) #Zero-pad color images
        print(self.histology_color)
        self.orig_hist_NIFTI = self.histology_color.convert_to_nifti(self.orig_hist_loc + '/orig/NIFTI','',self.overwrite)
        print(self.orig_hist_NIFTI)
        self.histology.segmentation(self.orig_hist_loc +"/segmented/orig",overwrite = self.overwrite) #GMM segmentation of color images 
        self.histology.zpad(self.orig_hist_loc + '/segmented/padded/orig',self.overwrite) #Z-pad segmented images separately (padding before segmenting throws off GMM)
        #Split RGB image into 3 separate channels
        self.histology.Blue_vol,self.histology.Green_vol,self.histology.Red_vol = self.color_split(self.histology_color,self.orig_col_split_loc + '/Red/orig',self.orig_col_split_loc + '/Green/orig',self.orig_col_split_loc + '/Blue/orig',self.overwrite)
        #Convert all preprocess files into NIFTI format
        self.hist_NIFTI = self.histology.convert_to_nifti(self.orig_hist_loc + '/segmented/padded/NIFTI','',self.overwrite)
        
        self.hist_NIFTI.Blue_vol = self.histology.Blue_vol.convert_to_nifti(self.orig_col_split_loc + '/Blue/NIFTI','',self.overwrite)
        self.hist_NIFTI.Green_vol = self.histology.Green_vol.convert_to_nifti(self.orig_col_split_loc + '/Green/NIFTI','',self.overwrite)
        self.hist_NIFTI.Red_vol = self.histology.Red_vol.convert_to_nifti(self.orig_col_split_loc + '/Red/NIFTI','',self.overwrite)
    def preprocess_blockface(self):
        """
        Converts blockface images to NIFTI format, pads the data and resamples NIFTI files to higher resolution if necessary.
        """
        print('Preprocessing BF')
        #Pad bf images
        self.bf.zpad(self.orig_bf_loc + '/orig/',self.overwrite)
        #Convert to NIFTI
        self.BF_NIFTI = self.bf.convert_to_nifti(self.orig_bf_loc + '/NIFTI/orig/','',self.overwrite)
        #Output volumetric bf NIFTI file for alignment to MRI
        self.BF_vol = self.BF_NIFTI.volumize(self.orig_bf_loc + '/volume/orig/blockface_vol.nii.gz',self.overwrite)
        self.BF_low_res_name = self.orig_bf_loc + '/volume/orig/blockface_vol_MRI_resample.nii.gz'
        if self.overwrite == False and os.path.isfile(self.BF_low_res_name):
            print(' - {} Already Exists. Utilizing currently existing data.'.format(self.BF_low_res_name))
        else:
            MRI_load = nib.load(self.MRI)
            self.resample(self.BF_vol,self.orig_bf_loc + '/volume/orig/blockface_vol_MRI_resample.nii.gz',MRI_load.header['pixdim'][1:4],MRI_load.affine,3)
    def preprocess_MRI(self):
        """
        Runs N4 Bias Correction and resamples MRI to higher resolution if necessary.
        """
        print('Preprocessing MRI')
        self.MRI_path = self.root_dir + 'MRI/'
        if self.input.MRI.N4 == True: 
            self.MRI_path += 'N4_'
            if os.path.isfile(self.MRI_path + os.path.split(self.MRI)[1]) and self.overwrite == False:
                print(' - ' + self.MRI_path + os.path.split(self.MRI)[1] + ' Already Exists')
                self.MRI = self.MRI_path + os.path.split(self.MRI)[1]
            else:
                #Run N4 Bias Field Correction on MRI Volume
                N4 = N4BiasFieldCorrection()
                N4.inputs.dimension = 3
                N4.inputs.input_image = self.MRI
                N4.inputs.output_image = self.MRI_path + os.path.split(self.MRI)[1]
                N4.cmdline
                N4.run()
                #Redefine MRI as the N4 corrected volume.
                self.MRI = N4.inputs.output_image
        self.MRI_path = self.MRI_path + 'reoriented_'
        if os.path.isfile(self.MRI_path + os.path.split(self.orig_MRI)[1]) and self.overwrite == False:
           print(' - ' + self.MRI_path + os.path.split(self.orig_MRI)[1] + ' Already Exists')
           self.MRI = self.MRI_path + os.path.split(self.orig_MRI)[1]
        else:
            res = afni.Resample()
            res.inputs.in_file = self.MRI
            tmp = self.histology.orientation
            #Resample MRI to the orientation found in the histology and blockface images
            #NOTE: AFNI (the program used below) and Nibabel (the program used to read/write NIFTI files)
            #use opposite orientation nomenclature. While confusing, this means the orientation string must
            #be inverted before using AFNI.
            if self.histology.orientation.find('S') >= 0:
                tmp.replace('S','I')
            else:
                tmp.replace('I','S')
            if self.histology.orientation.find('R') >= 0:
                tmp.replace('R','L')
            else:
                tmp.replace('L','R')
            if self.histology.orientation.find('P') >= 0:
                tmp.replace('P','A')
            else:
                tmp.replace('A','P')
            print(tmp)
            res.inputs.orientation = tmp
            res.inputs.out_file = self.MRI_path + os.path.split(self.orig_MRI)[1]
            res.run()
            self.MRI = res.inputs.out_file
        ##Skullstripping to come at a later time.
    def color_split(self, rgb_vol,out_R,out_G,out_B,overwrite):
        """
        Bypasses RGB restrictions in NIFTI by splitting the RGB channels in 
        the histology slices.
        Returns 3 IMAGE_Stacks, one for each color channel
        """
        #Define output directories
        out_R = os.path.abspath(out_R) + '/'
        out_G = os.path.abspath(out_G) + '/'
        out_B = os.path.abspath(out_B) + '/'
        break_flag = 0
        if overwrite == False:
            #Check if every file exists for each color channel
            for out in [out_R, out_G, out_B]:
                for i in range(len(rgb_vol.slices)):
                    if not os.path.isfile(out + rgb_vol.slices[i].name):
                        break_flag = 1
                        break
                if break_flag == 1:
                    break
            else:
                print(' - All Color Channel Split Files Exist. Utilizing currently existing data.')
                return Stacks.IMAGE_Stack(out_B,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim),Stacks.IMAGE_Stack(out_G,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim),Stacks.IMAGE_Stack(out_R,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim)
        for i in range(len(rgb_vol.slices)):
            #Split the RGB channels of each image
            print('Splitting Channels of {}'.format(rgb_vol.slices[i].name))
            im = plt.imread(rgb_vol.slices[i].path)
            print(im.shape)
            b, g, r = cv2.split(im[:,:,0:3])
            #Save split channels as individual images
            plt.imsave(out_B + rgb_vol.slices[i].name,b,cmap=plt.cm.gray)
            plt.imsave(out_R + rgb_vol.slices[i].name,r,cmap=plt.cm.gray)
            plt.imsave(out_G + rgb_vol.slices[i].name,g,cmap=plt.cm.gray)
        return Stacks.IMAGE_Stack(out_B,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim),Stacks.IMAGE_Stack(out_G,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim),Stacks.IMAGE_Stack(out_R,rgb_vol.pattern,rgb_vol.orientation,rgb_vol.pix_dim)
    def resample(self,in_vol,out,ref_vol_pixdim,ref_vol_affine,dim):
        """
        Resamples the in_vol to the resolution of the ref_vol.
        NOTE: This is an ANTs command that has not yet been programmed to run through Nipype.
        """
        print('Resampling {}'.format(in_vol))
        if dim == 2:
            #if 2d, resample the first 2 dimensions
            resample_command = "ResampleImage {0} {1} {2} {3}x{4}"
            resample_command = resample_command.format(dim,in_vol,out,ref_vol_pixdim[0], ref_vol_pixdim[1])
            in_vol.affine_3D = ref_vol_affine
        elif dim == 3:
            #if 3d, resample the first 3 dimensions
            resample_command = "ResampleImage {0} {1} {2} {3}x{4}x{5}"
            resample_command = resample_command.format(dim,in_vol,out,ref_vol_pixdim[0], ref_vol_pixdim[1], ref_vol_pixdim[2])
        os.system(resample_command)
#####################################################################################################################        
    def blockface_to_MRI_alignment(self,out_dir):
        """
        3D nonlinear alignment of the blockface NIFTI volume to the MRI reference volume.
        Alignment uses the program ANTs through the Nipype module.
        Outputs a 3D composite transform file that allows for warping of files from blockface space to MRI space.
        NOTE: VERY TIME CONSUMING STEP WHEN MRI HAS BEEN RESAMPLED TO BLOCKFACE OR HISTOLOGY RESOLUTION
        """
        out_dir = os.path.abspath(out_dir) + '/'
        if self.overwrite == False and os.path.isfile(out_dir + 'MRI_to_blockface_linear_alignment.nii.gz'):
            print(' - Reference MRI Already Linearly Aligned to Blockface')
            return
        else:
            out_dir = os.path.abspath(out_dir) + '/'
            #Define inputs to nipype's ANTs Registration.
            reg = Registration()
            reg.inputs.fixed_image = self.BF_low_res_name
            reg.inputs.moving_image = self.MRI
            reg.inputs.output_warped_image = out_dir + 'MRI_to_blockface_linear_alignment.nii.gz'
            reg.inputs.output_transform_prefix = out_dir + "composite_transform_MRI_to_blockface_alignment_linear"
            reg.inputs.transforms = ['Translation','Rigid']
            reg.inputs.transform_parameters = [(0.1,), (0.1,)]
            reg.inputs.number_of_iterations = ([[1500,500,250]] * 2)
            reg.inputs.interpolation='BSpline'
            reg.inputs.dimension = 3
            reg.inputs.write_composite_transform = True
            reg.inputs.collapse_output_transforms = True
            reg.inputs.metric = ['Mattes'] * 2
            reg.inputs.metric_weight = [1] * 2
            reg.inputs.radius_or_number_of_bins = [32] * 2
            reg.inputs.sampling_strategy = ['Regular'] * 2
            reg.inputs.sampling_percentage = [0.3] * 2
            reg.inputs.convergence_threshold = [1.e-6] * 2
            reg.inputs.convergence_window_size = [20] * 2
            reg.inputs.smoothing_sigmas = [[0, 0, 0]] * 2
            reg.inputs.sigma_units = ['vox'] * 2
            reg.inputs.shrink_factors = [[3, 2, 1]] * 2
            reg.inputs.use_estimate_learning_rate_once = [True] * 2
            reg.inputs.use_histogram_matching = [False] * 2
            reg.inputs.initial_moving_transform_com = True
            reg.inputs.num_threads = self.threads
            reg.inputs.verbose = True
            print(reg.cmdline)
            reg.run()
            #Define inputs to nipype's ANTs Registration.
        if self.overwrite == False and os.path.isfile(out_dir + 'blockface_to_MRI_alignment.nii.gz'):
            print(' - Blockface images already aligned to reference MRI')
            return
        else:
            reg = Registration()
            reg.inputs.fixed_image = out_dir + 'MRI_to_blockface_linear_alignment.nii.gz'
            reg.inputs.moving_image = self.BF_vol
            reg.inputs.output_warped_image = out_dir + 'blockface_to_MRI_alignment.nii.gz'
            reg.inputs.output_transform_prefix = out_dir + "composite_transform_blockface_to_MRI_alignment"
            reg.inputs.transforms = ['Affine','SyN']
            reg.inputs.transform_parameters = [(0.1,), (0.1,)]
            reg.inputs.number_of_iterations = ([[1500,500,250]] * 2)
            reg.inputs.interpolation='BSpline'
            reg.inputs.dimension = 3
            reg.inputs.write_composite_transform = True
            reg.inputs.collapse_output_transforms = True
            reg.inputs.metric = ['Mattes'] * 2
            reg.inputs.metric_weight = [1] * 2
            reg.inputs.radius_or_number_of_bins = [32] * 2
            reg.inputs.sampling_strategy = ['Regular'] * 2
            reg.inputs.sampling_percentage = [0.3] * 2
            reg.inputs.convergence_threshold = [1.e-6] * 2
            reg.inputs.convergence_window_size = [20] * 2
            reg.inputs.smoothing_sigmas = [[0, 0, 0]] * 2
            reg.inputs.sigma_units = ['vox'] * 2
            reg.inputs.shrink_factors = [[3, 2, 1]] * 2
            reg.inputs.use_estimate_learning_rate_once = [True] * 2
            reg.inputs.use_histogram_matching = [False] * 2
            reg.inputs.initial_moving_transform_com = True
            reg.inputs.num_threads = self.threads
            reg.inputs.verbose = True
            print(reg.cmdline)
            reg.run()
            return
    def slice_by_slice_alignment(self,threads,out_dir):
        """
        2D alignment of each histology slice to a corresponding blockface slice using 
        ANTs registration tools through Nipype. Outputs histology NIFTI files that have
        been aligned to the blockface images.
        Because each alignment is independent of the other, the script will utilize all
        threads provided by the user to align multiple slices simultaneously.
        See Register_Wrapper for more information and the alignment code.
        """
        if len(self.BF_NIFTI.slices) != len(self.hist_NIFTI.slices):
            #Check to make sure volumes are of identical size
            print('Volumes are different lengths, cannot align')
            return
        elif self.overwrite == False:
            #Check if output files already exist
            for i in range(len(self.hist_NIFTI.slices)):
                slice_num = ''
                if i < 10:
                    slice_num = '000' + str(i)
                elif i < 100:
                    slice_num = '00' + str(i)
                elif i < 1000:
                    slice_num = '0' + str(i)
                elif slice < 10000:
                    slice_num = str(i) 
                if not os.path.isfile(out_dir + '/composite_transform/composite_transform_{}.h5'.format(slice_num)) or not os.path.isfile(out_dir + '/grayscale/Hist_to_BF_{}.nii.gz'.format(slice_num)):
                    break
            else:
                print(' - All Aligned NIFTI Files and Transformation Matrices Exist. Utilizing currently existing data.')
                self.hist_transform= Stacks.Stack(out_dir + '/composite_transform/','composite_transform_**.h5')
                self.aligned_histology = Stacks.NIFTI_Stack(out_dir + '/grayscale/','Hist_to_BF**.nii.gz')
                return
        print('====================================ATTEMPTING TO MULTITHREAD====================================')
        #Feed alignment information to sub-processes through Register_Wrapper
        pool = Pool(processes=self.threads)
        pool.map(Register_Wrapper(self.BF_NIFTI,self.hist_NIFTI,out_dir,self.reg_method), list(range(len(self.BF_NIFTI.slices))))
        pool.close()
        pool.join()
        self.hist_transform= Stacks.Stack(out_dir + '/composite_transform/','composite_transform_**.h5')
        self.aligned_histology = Stacks.NIFTI_Stack(out_dir + '/grayscale/','Hist_to_BF**.nii.gz')
    def colorize(self,col_out_dir,final_out_dir):
        """
        Transforms the individual channels from Pipeline.color_split using the 
        affine/nonlinear transformation parameters from Pipeline.slice_by_slice_alignment() 
        and the nonlinear volumetric transformation parameters from Pipeline.blockface_to_MRI_alignment()
        Because each transformation is independent of the others, the script will utilize all
        threads provided by the user to transform multiple slices simultaneously.
        See Transform_Wrapper for more information and the transformation code.
        """
        #Feed transformation information to sub-processes through Transform_Wrapper
        out_suf_list = ['Blue','Green','Red']
        skip_flag = False
        for i, col_vol in enumerate([self.hist_NIFTI.Blue_vol, self.hist_NIFTI.Green_vol, self.hist_NIFTI.Red_vol]):
            for j in range(len(self.hist_NIFTI.slices)):
                if not os.path.isfile(self.orig_slice_by_slice_loc +'/color/' + out_suf_list[i] + '/' + col_vol.slices[j].name):
                    break
            else:
                continue
            break
        else:
            print(' - All Color Channel Split Transformed Files Exist. Utilizing currently existing data.')
            skip_flag = True
        if skip_flag == False:
            print('====================================ATTEMPTING TO MULTITHREAD====================================')
            pool = Pool(processes=self.threads)
            self.hist_NIFTI.Blue_vol.col = 'Blue'
            self.hist_NIFTI.Green_vol.col = 'Green'
            self.hist_NIFTI.Red_vol.col = 'Red'
            for col_vol in enumerate([self.hist_NIFTI.Blue_vol, self.hist_NIFTI.Green_vol, self.hist_NIFTI.Red_vol]):
                pool.map(Transform_Wrapper(col_vol,self.hist_transform,self.BF_NIFTI,self.orig_slice_by_slice_loc +'/color/'), list(range(len(self.hist_transform.slices))))
                pool.close()
                pool.join()
        #Load output color channel split Stacks and convert to volume/
        tmp = self.BF_NIFTI
        r = Stacks.NIFTI_Stack(self.orig_slice_by_slice_loc +'/color/Red/')
        r.affine_3D = tmp.affine_3D
        r.volumize(self.orig_slice_by_slice_loc +'/color/volumes/r_vol.nii.gz')
        g = Stacks.NIFTI_Stack(self.orig_slice_by_slice_loc +'/color/Green/')
        g.affine_3D = tmp.affine_3Dblockface_to_MRI_alignment.nii.gz
        g.volumize(self.orig_slice_by_slice_loc +'/color/volumes/g_vol.nii.gz')
        b = Stacks.NIFTI_Stack(self.orig_slice_by_slice_loc +'/color/Blue/')
        b.affine_3D = tmp.affine_3D
        b.volumize(self.orig_slice_by_slice_loc +'/color/volumes/b_vol.nii.gz')
        #Transform color-split volumes to the MRI space
        self.final_apply_transform(self.orig_slice_by_slice_loc +'/color/volumes/b_vol.nii.gz',self.orig_slice_by_slice_loc +'/color/volumes/final_b_vol.nii.gz')
        self.final_apply_transform(self.orig_slice_by_slice_loc +'/color/volumes/g_vol.nii.gz',self.orig_slice_by_slice_loc +'/color/volumes/final_g_vol.nii.gz')
        self.final_apply_transform(self.orig_slice_by_slice_loc +'/color/volumes/r_vol.nii.gz',self.orig_slice_by_slice_loc +'/color/volumes/final_r_vol.nii.gz')
        #Load transformed and color-split volumes. Merge the channels to create and RGB volume.
        print('Loading RGB')
        r_data = nib.load(self.orig_slice_by_slice_loc +'/color/volumes/final_r_vol.nii.gz' ).get_data()
        g_data = nib.load(self.orig_slice_by_slice_loc +'/color/volumes/final_g_vol.nii.gz').get_data()
        b_data = nib.load(self.orig_slice_by_slice_loc +'/color/volumes/final_b_vol.nii.gz').get_data()
        print('Merging Channels')
        rgb =  np.empty((r_data.shape[0],r_data.shape[1],r_data.shape[2],3))
        rgb[:,:,:,0] =b_data
        rgb[:,:,:,1] =g_data
        rgb[:,:,:,2] =r_data
        rgb = rgb.astype('u1')
        #Save the RGB Volume
        print('Saving Volume')
        shape_3d = rgb.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        rgb_typed = rgb.view(rgb_dtype).reshape(shape_3d)
        tmp = nib.load(self.MRI)
        volume = nib.Nifti1Image(rgb_typed, affine=tmp.affine)
        nib.save(volume,final_out_dir + '/RGB_aligned_histology_vol.nii.gz')
        #volume = nib.Nifti1Image(color.rgb2grey(rgb_typed), affine=tmp.affine)
        #nib.save(volume,final_out_dir + '/greyscale_aligned_histology_vol.nii.gz')
    
    
    def final_apply_transform(self,in_file,out_file):
        """
        Uses ANTs registration tools through Nipype to warp the split channel RGB volumes into 
        an MRI reference space.
        """
        at = ApplyTransforms()
        at.inputs.dimension = 3
        at.inputs.input_image = in_file
        at.inputs.reference_image = self.orig_bf_loc + '/volume/aligned_to_MRI/blockface_to_MRI_alignment.nii.gz'#self.MRI#self.MRI_path + os.path.split(self.orig_MRI)[1]
        at.inputs.transforms = self.orig_bf_loc + '/volume/aligned_to_MRI/composite_transform_blockface_to_MRI_alignmentComposite.h5'
        at.inputs.interpolation = 'BSpline'
        at.inputs.output_image = out_file
        at.inputs.invert_transform_flags = [False]
        at.inputs.interpolation_parameters = (5,)
        print(at.cmdline)
        at.run()
###############################################################
class Register_Wrapper(object):
    """
    Wrapper class for 2D registration of objects. 
    Created to be fed into a multiprocessing pool.
    Only call from Pipeline.slice_by_slice_alignment() to avoid errors. 
    """
    def __init__(self, BF_vol,Hist_vol,out_dir,reg_method):
        """
        Defines static inputs for all iterations of the transformation.
        """
        self.BF_vol = BF_vol
        self.Hist_vol = Hist_vol
        self.out_dir = out_dir
        self.reg_method = reg_method
    def __call__(self, i):
        """
        Uses static and iterative input variables to call register().
        """
        self.register(i)
    def register(self,i):
        """
        Aligns a 2D histology image to a 2D blockface image using ANTs 
        registration tools.
        """
        #Create naming convention for aligned files.
        slice_num = ''
        if i < 10:
            slice_num = '000' + str(i)
        elif i < 100:
            slice_num = '00' + str(i)
        elif i < 1000:
            slice_num = '0' + str(i)
        elif slice < 10000:
            slice_num = str(i)   
        #Define registration parameters for ANT's Registration command through Nipype.
        reg = Registration()
        #reg.inputs.verbose = True
        reg.inputs.fixed_image = self.BF_vol.slices[i].path
        reg.inputs.moving_image = self.Hist_vol.slices[i].path
        reg.inputs.output_warped_image = 'Hist_to_BF_{}.nii.gz'.format(slice_num)
        reg.inputs.output_transform_prefix = "composite_transform_{}.h5".format(slice_num)
        if self.reg_method == 'nonlinear':
            reg.inputs.transforms = ['Translation', 'Rigid', 'Affine','SyN']
            reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1,), (0.1,)]
            reg.inputs.number_of_iterations = ([[1500,500,250]] * 4)
            reg.inputs.metric = ['Mattes'] * 4
            reg.inputs.metric_weight = [1] * 4
            reg.inputs.radius_or_number_of_bins = [32] * 4
            reg.inputs.sampling_strategy = ['Regular'] * 4
            reg.inputs.sampling_percentage = [0.3] * 4
            reg.inputs.convergence_threshold = [1.e-6] * 4
            reg.inputs.convergence_window_size = [20] * 4
            reg.inputs.smoothing_sigmas = [[0, 0, 0]] * 4
            reg.inputs.sigma_units = ['vox'] * 4
            reg.inputs.shrink_factors = [[6, 4, 2]] + [[3, 2, 1]] * 3
            reg.inputs.use_estimate_learning_rate_once = [True] * 4
            reg.inputs.use_histogram_matching = [False] * 4
        else:
            if self.reg_method != 'linear':
                warnings.warn("Can't Interpret registration method, Defaulting to linear alignment.")
            reg.inputs.transforms = ['Translation', 'Rigid', 'Affine']
            reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1,)]
            reg.inputs.number_of_iterations = ([[1500,500,250]] * 3)
            reg.inputs.metric = ['Mattes'] * 3
            reg.inputs.metric_weight = [1] * 3
            reg.inputs.radius_or_number_of_bins = [32] * 3
            reg.inputs.sampling_strategy = ['Regular'] * 3
            reg.inputs.sampling_percentage = [0.3] * 3
            reg.inputs.convergence_threshold = [1.e-6] * 3
            reg.inputs.convergence_window_size = [20] * 3
            reg.inputs.smoothing_sigmas = [[0, 0, 0]] * 3
            reg.inputs.sigma_units = ['vox'] * 3
            reg.inputs.shrink_factors = [[3, 2, 1]] * 3
            reg.inputs.use_estimate_learning_rate_once = [True] * 3
            reg.inputs.use_histogram_matching = [False] * 3    
        reg.inputs.interpolation = 'BSpline'
        reg.inputs.dimension = 2
        reg.inputs.write_composite_transform = True
        reg.inputs.collapse_output_transforms = True
        reg.inputs.initial_moving_transform_com = True
        reg.inputs.float = True
        reg.inputs.ignore_exception = True
        outputs = reg._list_outputs()
        #print(reg.cmdline)
        #Copy output files to output directory.
        print("##################################",'Hist_to_BF_{}.nii.gz is RUNNING'.format(slice_num),"##################################\n")
        reg.run()
        shutil.move(outputs['warped_image'],self.out_dir + '/grayscale/Hist_to_BF_{}.nii.gz'.format(slice_num))
        shutil.move(outputs['composite_transform'],self.out_dir + '/composite_transform/composite_transform_{}.h5'.format(slice_num))
        print("##################################",'Hist_to_BF_{}.nii.gz is COMPLETE'.format(slice_num),"##################################\n")

class Transform_Wrapper(object):
    """
    Wrapper class for 2D registration of objects. 
    Created to be fed into a multiprocessing pool.
    Only call from Pipeline.colorize() to avoid errors. 
    Transforms 2D images using affine transformation parameters from ANTs
    """
    def __init__(self, vol,hist_transform,BF_NIFTI,out_dir):
        """
        Defines static inputs for all iterations of the transformation.
        """
        self.vol = vol
        self.hist_transform = hist_transform
        self.BF_NIFTI = BF_NIFTI
        self.out_dir = os.path.abspath(out_dir) + '/'
    def __call__(self, i):
        """
        Uses static and iterative input variables to call apply_transform_slice() for each color channel.
        """
        self.apply_transform_slice(i,self.vol,self.vol.col)
    def apply_transform_slice(self,i,col_vol,out_suf):
        """
        Applies a previously calculated transformation (self.hist_transform) to a given color channel.
        """
        #Define input variables for ANT's ApplyTransform command through Nipype
        at = ApplyTransforms()
        at.inputs.dimension = 2
        at.inputs.input_image = col_vol.slices[i].path
        at.inputs.reference_image = self.BF_NIFTI.slices[i].path
        at.inputs.transforms = self.hist_transform.slices[i].path
        at.inputs.interpolation = 'BSpline'
        at.inputs.output_image = self.out_dir + out_suf + '/' + col_vol.slices[i].name
        at.inputs.invert_transform_flags = [False]
        at.inputs.interpolation_parameters = (5,)
        print(at.cmdline)
        at.run()
        