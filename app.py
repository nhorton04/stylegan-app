import numpy as np
import os
import pickle
import streamlit as st
import tensorflow as tf

from align_images import unpack_bz2, align, LANDMARKS_MODEL_URL
from ffhq_dataset.face_alignment import image_align
# from ffhq_dataset.landmarks_detector import LandmarksDetector, get_landmarks
from keras.utils import get_file
import bz2
import dlib
import config
import dnnlib.tflib as tflib
import PIL.Image
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input


def main():

    uploaded_file = st.file_uploader(
        "Choose first picture", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
    else:
        uploaded_file = os.path.abspath(os.getcwd()) + '/images/Gunny.jpg'
        st.image(uploaded_file, width=50)

    second_upload = st.file_uploader(
        "Choose second picture", type=['jpg', 'jpeg', 'png'])
    if second_upload is not None:
        st.image(second_upload, width=200)
    else:
        second_upload = os.path.abspath(os.getcwd()) + '/images/me.png'
        st.image(second_upload, width=50)

    images = [uploaded_file, second_upload]

    ALIGNED_IMAGES_DIR = os.getcwd() + '/aligned'
    RAW_IMAGES_DIR = os.getcwd() + '/images'

    align()


if __name__ == "__main__":
    main()
