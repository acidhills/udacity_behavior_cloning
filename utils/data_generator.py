import cv2
from sklearn.utils import shuffle
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def get_generator(directories, batch_size=32):
    train_generators = []
    valid_generators = []
    train_len, valid_len = 0, 0
    for dir in directories:
        samples = []
        with open(dir + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                samples.append(line)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)
        train_len += len(train_samples)
        valid_len += len(validation_samples)

        train_generator = generator(train_samples, dir, batch_size=batch_size)
        validation_generator = generator(validation_samples, dir, batch_size=batch_size)
        train_generators.append(train_generator)
        valid_generators.append(validation_generator)
    return shuffle_gens_(train_generators), shuffle_gens_(valid_generators), train_len, valid_len


def shuffle_gens_(generators):
    valid_gens = {}
    for gen in generators:
        valid_gens[gen] = True
    while True:
        for gen in generators:
            if valid_gens[gen]:
                res, x, y = get_val_(gen)
                if not res:
                    valid_gens[gen] = False
                    continue
                yield x, y


def get_val_(gen):
    try:
        x, y = next(gen)
        return True, x, y
    except StopIteration:
        return False, None, None





def generator(samples, data_directory, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = data_directory + 'IMG/' + batch_sample[0].split('/')[-1]
                left_name = data_directory + 'IMG/' + batch_sample[1].split('/')[-1]
                right_name = data_directory + 'IMG/' + batch_sample[2].split('/')[-1]

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

            X_train, y_train = add_flip_(X_train, y_train)
            X_train, y_train = add_darkness_(X_train, y_train)

            yield shuffle(X_train, y_train)


def add_flip_(x, y):
    x = np.concatenate((x, x[..., ::-1, :]))
    y = np.concatenate((y, -y))
    return x, y


def add_darkness_(x,y):
    beta = np.random.uniform(-120, -50)
    colored = [cv2.convertScaleAbs(img, beta=beta) for img in x]
    x = np.concatenate((x, colored))
    y_train = np.concatenate((y, y))
    return  x, y_train
