# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:37:09 2017

@author: benju
"""
import Stacks
import os
class InputSpec():
    """
    Class designed to hold input variables and documentation until the Pipeline class
    is ready to run.
    """
    def __init__(self):
        """Initiates InputSpec Class. Holds variables for initialization of pipeline."""
        self.histology = InputVar('')
        self.bf = InputVar('')
        self.MRI = InputVar('')
        self.histology.dir = None
        self.bf.dir = None
        self.root_dir = None
        self.histology.orientation = 'RAI'
        self.histology.pix_dim = [1,1,1]
        self.histology.pattern = '*'
        self.bf.orientation = 'RAI'
        self.bf.pix_dim = [1,1,1]
        self.bf.pattern = '*'
        self.MRI.N4 = True
        self.MRI.skullstrip = False
        self.overwrite = False
        self.threads = 1
        self.reg_method = 'linear'
        self.resolution_level = 'MRI'
        self.color = True
        self.document_inputs()
    def document_inputs(self):
        """Generates InputVarHelp objects that contain documentation for all potential pipeline variables."""
        self.required = []
        required_in = [['histology.dir','Directory containing all histology images','/data/image_storage/histology_slices/',self.histology.dir],
                       ['bf.dir','Directory containing all blockface images','/data/image_storage/blockface_slices/',self.bf.dir],
                       ['MRI.name','An MRI of the subject','monkey.nii.gz',self.MRI],
                       ['root_dir','The output directory','/data/image_storage/final_volume/',self.root_dir]]
        for i in range(len(required_in)):
            self.required.append(InputVarHelp(required_in[i][0],required_in[i][1],required_in[i][2],required_in[i][3]))
        self.recommended = []
        recommended_in = [['histology.orientation','XYZ orientation of the image.  See NIFTI documentation for description of NIFTI orientation.','RAS',self.histology.orientation],
                           ['histology.pix_dim','XYZ dimensions of the pixels in mm','[0.02, 0.02,0.250]',self.histology.pix_dim],
                           ['bf.orientation','XYZ orientation of the image. See NIFTI documentation for description of NIFTI orientation.','RAS',self.bf.orientation],
                           ['bf.pix_dim','XYZ dimensions of the pixels in mm','[0.05,0.05,0.250]',self.bf.pix_dim]]
        for i in range(len(recommended_in)):
            self.recommended.append(InputVarHelp(recommended_in[i][0],recommended_in[i][1],recommended_in[i][2],recommended_in[i][3]))
        self.optional = []
        optional_in = [['histology.pattern','Only includes files in histology stack that contain Unix-style pattern','hist_**.jpg',self.histology.pattern],
                       ['bf.pattern','Only includes files in blockface Stack that contain Unix-style pattern','bf_**.png',self.bf.pattern],
                       ['MRI.N4','Perform N4 Bias Correction',True,self.MRI.N4],
                       ['MRI.skullstrip','Perform MRI Skullstripping. NOTE: Future feature. Not currently implemented.',True,self.MRI.skullstrip],
                       ['overwrite','Overwrite Previously Stored Results. If False, program will attempt to use previously stored intermediate files.',False,self.overwrite],
                       ['threads','Number of processes available for multiprocessing',1,self.threads],
                       ['2D_reg_method','Use linear or nonlinear warping for 2D slice alignment?',"'linear'|'nonlinear'",self.reg_method],
                       ['resolution_level','Final resolution of the histology volume. WARNING: blockface and histology settings are associated with obscene processing requirements.',"'MRI'|'blockface'|'histology'",self.reg_method],
                       ['color','Whether or not to have a color image output',True,self.color]]
        for i in range(len(optional_in)):
            self.optional.append(InputVarHelp(optional_in[i][0],optional_in[i][1],optional_in[i][2],optional_in[i][3]))
    def print_help(self):
        """Print documentation related to input variables"""
        print('Required Inputs:')
        for i in self.required:
            print('\t',i.name,': ',i.desc)
            print('\t\t Example: ',i.ex)
        print('Recommended Inputs:')
        for i in self.recommended:
            print('\t',i.name,': ',i.desc)
            print('\t\t Example: ',i.ex)
        print('Optional Inputs:')
        for i in self.optional:
            print('\t',i.name,': ',i.desc)
            print('\t\t Example: ',i.ex)
    def print_inputs(self):
        """Display stored variable values for all variables"""
        self.document_inputs()
        print('Required Inputs:')
        for i in self.required:
            print('\t',i.name,'= ',i.value)
        print('Recommended Inputs:')
        for i in self.recommended:
            print('\t',i.name,'= ',i.value)
        print('Optional Inputs:')
        for i in self.optional:
            print('\t',i.name,'= ',i.value)
    def check_inputs(self):
        """Returns True if all required variables are user defined. Returns false otherwise."""
        for i in self.required:
            if eval('self.' + i.name) is None:
                break
        else:
            return True
        return False
    def initialize_inputs(self):
        """Uses stored variable input data to define Pipeline volumes and variables if required inputs have been user defined."""
        validate = self.check_inputs()
        if validate == True:
            bf_out = Stacks.IMAGE_Stack(self.bf.dir,self.bf.pattern,orientation = self.bf.orientation, pix_dim = self.bf.pix_dim)
            histology_out = Stacks.IMAGE_Stack(self.histology.dir,self.histology.pattern,orientation = self.histology.orientation, pix_dim = self.histology.pix_dim)
            MRI_out = self.MRI.name
            out_dir_out = os.path.abspath(self.root_dir) + '/'
            return (bf_out, histology_out,MRI_out,out_dir_out,self.overwrite, self.threads,self.reg_method, self.resolution_level, self.color)
        else:
            print('All Required Variables Must be Defined')
            self.print_help()
            return None, None, None, None
class InputVarHelp():
    """Documentation object for input variables."""
    def __init__(self,name,desc,example='',value=None):
        self.name = name
        self.desc = desc
        self.ex = example
        self.value = value
class InputVar():
    """Temporary object meant to store user-defined object attributes without creating the objects."""
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return self.name