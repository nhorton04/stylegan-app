import numpy as np
import os
import pickle
import streamlit as st
import tensorflow as tf

from align_images import unpack_bz2, align, LANDMARKS_MODEL_URL
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector, get_landmarks

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

    second_upload = st.file_uploader(
        "Choose second picture", type=['jpg', 'jpeg', 'png'])
    if second_upload is not None:
        st.image(second_upload, width=200)

    images = [uploaded_file, second_upload]

    for img_name in images:
        print('Aligning %s ...' % img_name)
        try:
            # raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            # fn = face_img_name = '%s_%02d.png' % (
            #     os.path.splitext(img_name)[0], 1)
            # st.write(fn)
            # if os.path.isfile(fn):
            #     continue
            print('Getting landmarks...')
            # get_landmarks(img_name)

            # for i, face_landmarks in enumerate(get_landmarks(img_name), start=1):
            try:
                print('Starting face alignment...')
                face_img_name = f'{img_name.name}.png'
                print("awwagga")
                print(f'imageeee ==== {face_img_name}')
                aligned_face_path = os.path.join(
                    ALIGNED_IMAGES_DIR, face_img_name)
                image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=args.output_size,
                            x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
                print('Wrote result %s' % aligned_face_path)
            except:
                print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")


if __name__ == "__main__":
    main()
