import numpy as np
import os
import pickle
class DataProcessor:
    @staticmethod
    def load_data():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        text_path = os.path.join(current_dir, 'input/text.pickle')
        audio_path = os.path.join(current_dir, 'input/audio.pickle')
        video_path = os.path.join(current_dir, 'input/video.pickle')

        # Load the datasets
        (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(open(text_path, 'rb'))
        (train_audio, _, test_audio, _, _, _, _) = pickle.load(open(audio_path, 'rb'))
        (train_video, _, test_video, _, _, _, _) = pickle.load(open(video_path, 'rb'))

        # Return all the data loaded
        return (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len,
                train_audio, test_audio, train_video, test_video)


    @staticmethod
    def create_one_hot_labels(train_label, test_label):
        """
        Converts train and test labels into one-hot encoded format.

        Args:
            train_label (np.array): 2D array of train labels.
            test_label (np.array): 2D array of test labels.

        Returns:
            tuple: 3D arrays of one-hot encoded train and test labels.
        """
        maxlen = int(max(train_label.max(), test_label.max()))
        train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
        test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

        for i in range(train_label.shape[0]):
            for j in range(train_label.shape[1]):
                train[i, j, train_label[i, j]] = 1

        for i in range(test_label.shape[0]):
            for j in range(test_label.shape[1]):
                test[i, j, test_label[i, j]] = 1

        return train, test

    @staticmethod
    def create_mask(train_data, test_data, train_length, test_length):
        """
        Creates masks for training and testing data based on their lengths.

        Args:
            train_data (np.array): 2D array of train data.
            test_data (np.array): 2D array of test data.
            train_length (list): List of lengths for each train sample.
            test_length (list): List of lengths for each test sample.

        Returns:
            tuple: 2D arrays of masks for train and test data.
        """
        train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
        for i in range(len(train_length)):
            train_mask[i, :train_length[i]] = 1.0

        test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
        for i in range(len(test_length)):
            test_mask[i, :test_length[i]] = 1.0

        return train_mask, test_mask
    
        
    @staticmethod
    def split_dataset(data, train_ratio=0.8):
        """
        Split data into training and development (validation) sets.
        :param data: The data to be split.
        :param train_ratio: Ratio of data to be used as training data.
        :return: A tuple of (training_data, development_data).
        """
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
