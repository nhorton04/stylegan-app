import dlib
from keras.utils import get_file
import bz2


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        print('step1')
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)
        print('step2')
        for detection in dets:
            try:
                face_landmarks = [
                    (item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                print('step3')
                yield face_landmarks
                print('step4')
            except:
                print("Exception in get_landmarks()!")


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))

detector = dlib.get_frontal_face_detector(
)  # cnn_face_detection_model_v1 also can be used
shape_predictor = dlib.shape_predictor(landmarks_model_path)


def get_landmarks(image):
    print('step1')
    img = dlib.load_rgb_image(image)
    dets = detector(img, 1)
    print('step2')
    for detection in dets:
        try:
            face_landmarks = [
                (item.x, item.y) for item in shape_predictor(img, detection).parts()]
            print('step3')
            yield face_landmarks
            print('step4')
        except:
            print("Exception in get_landmarks()!")
