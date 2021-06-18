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

    second_upload = st.file_uploader(
        "Choose second picture", type=['jpg', 'jpeg', 'png'])
    if second_upload is not None:
        st.image(second_upload, width=200)

    images = [uploaded_file, second_upload]

    ALIGNED_IMAGES_DIR = os.getcwd() + '/aligned'

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
            print('....')
            face_landmarks = get_landmarks(img_name)
            for landmark in face_landmarks:
                print(landmark)
            # get_landmarks(img_name)
            print('awaaga')
            # for i, face_landmarks in enumerate(get_landmarks(img_name), start=1):
            try:
                print('Starting face alignment...')
                face_img_name = f'{img_name.name}'
                print(f'imageeee ==== {face_img_name}')
                aligned_face_path = os.path.join(
                    ALIGNED_IMAGES_DIR, face_img_name)
                print(aligned_face_path)
                image_align(img_name, aligned_face_path, face_landmarks)
                # image_align(img_name, aligned_face_path, face_landmarks, output_size=1024,
                #             x_scale=1, y_scale=1, em_scale=0.1, alpha=False)
                print('Wrote result %s' % aligned_face_path)
            except:
                print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")


def get_landmarks(image):
    print('stpe1')
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    detector = dlib.get_frontal_face_detector(
    )  # cnn_face_detection_model_v1 also can be used
    # detector = dlib.get_frontal_face_detector(
    # )  # cnn_face_detection_model_v1 also can be used
    shape_predictor = dlib.shape_predictor(landmarks_model_path)
    print('step1')
    img = PIL.Image.open(image).convert('RGBA').convert('RGB')
    print('stpe?')
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    print('step2')
    for detection in dets:
        try:
            face_landmarks = [
                (item.x, item.y) for item in shape_predictor(img, detection).parts()]
            yield 'step3'
            yield face_landmarks
            print('step4')
        except:
            print("Exception in get_landmarks()!")


if __name__ == "__main__":
    main()
