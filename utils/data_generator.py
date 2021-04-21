import cv2
from sklearn.utils import shuffle
import numpy as np

def generator(samples, data_directory,batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name = data_directory + '/IMG/'+batch_sample[0].split('/')[-1]
                center_name = data_directory + batch_sample[0].strip()
                left_name = data_directory + batch_sample[1].strip()
                right_name = data_directory + batch_sample[2].strip()

                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)

                angle_mod = 0.2

                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + angle_mod
                right_angle = float(batch_sample[3]) - angle_mod

                images.append(center_image)
                angles.append(center_angle)

                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
