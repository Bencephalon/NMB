#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 08:20:34 2017

@author: jungbt
"""
import os
import sys
import hist_pipeline_gray as pip
sys.path.append(os.getcwd()) 
pipe = pip.Pipeline()
#pipe.run()
pipe.input.histology.dir = '/data/NIMH_LBC_49ers/NIMH_macaque_brain/Subjects/gb8A/GB8A_Thionin/'
pipe.input.bf.dir = '/data/NIMH_LBC_49ers/NIMH_macaque_brain/Subjects/gb8A/GB8A_Blockface/blockface_volume/alignment_2/manual_tracing_output'
pipe.input.MRI.name = '/data/NIMH_LBC_49ers/NMT_D99_test/GB8A_to_NMT/GB8A_SS/GB8A_T1_SS_re.nii.gz'
pipe.input.root_dir = '/data/NIMH_LBC_49ers/NIMH_macaque_brain/Subjects/gb8A/pipeline_test_final_high3'
pipe.input.histology.pattern = 'GB8A_**.jpg'
pipe.input.bf.pattern = 'GB8A_**.jpg'
pipe.input.histology.orientation = 'SRP'
pipe.input.bf.orientation = 'SRP'
pipe.input.histology.pix_dim = [0.025,0.025,0.250]
pipe.input.bf.pix_dim = [0.03,0.03,0.250]
pipe.input.overwrite = False
pipe.input.threads = 181
pipe.input.reg_method = 'nonlinear'
pipe.input.resolution_level = 'MRI'
pipe.input.color = False
pipe.input.print_inputs()
pipe.run()