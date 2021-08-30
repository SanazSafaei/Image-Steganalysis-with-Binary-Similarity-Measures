import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import random


class Svm:

    def __init__(self, i):
        # self.stego_dataset = stego_dataset
        # self.original_dataset = original_dataset
        self.train_input = []
        self.train_label = []
        self.test_input = []
        self.test_label = []
        self.i = i

    def initial_data(self):
        with open('features/train_input' + str(self.i) + '.txt', 'r') as filehandle:
            file = filehandle.read()
            line = file.replace("[", "")
            line = line.replace("]", "")
            list_line = list(line.split("|"))

            currentPlace = []
            for arr in list_line:
                l = []
                array = list(arr.split(" "))
                for i in array:
                    if i != '':
                        l.append(np.float(i))
                if (len(l) == 7):
                    currentPlace.append(l)

            self.train_input = currentPlace
        with open('features/train_label' + str(self.i) + '.txt', 'r') as filehandle:
            file = filehandle.read()
            # currentPlace = line[:-1]
            l = list(file.split("|"))
            self.train_label = l[:-1]

        with open('features/test_input' + str(self.i) + '.txt', 'r') as filehandle:
            file = filehandle.read()

            line = file.replace("[", "")
            line = line.replace("]", "")
            list_line = list(line.split("|"))
            currentPlace = []
            for arr in list_line:
                l = []
                array = list(arr.split(" "))
                for i in array:
                    if (i != ''):
                        l.append(np.float(i))
                if (len(l) == 7):
                    currentPlace.append(l)

            self.test_input = (currentPlace)

        with open('features/test_label' + str(self.i) + '.txt', 'r') as filehandle:
            file = filehandle.read()
            # currentPlace = line[:-1]
            l = list(file.split("|"))
            self.test_label = l[:-1]  # remove eof

    def svm(self):
        self.initial_data()
        print("input svm data : ", len(self.train_input), len(self.train_input[0]))
        classifier_poly2 = SVC(kernel='poly', gamma=0.3, degree=2, max_iter=1e5)
        file_name_poly2 = 'svm/saved_svm_kfold_poly2_scale_0_3' + str(self.i) + '.sav'
        classifier_linear = SVC(kernel='linear', gamma=0.22)
        file_name_linear = 'svm/saved_svm_kfold_linear_scale_0_22' + str(self.i) + '.sav'
        classifier_poly3 = SVC(kernel='poly', gamma=0.22, degree=3, max_iter=1e5)
        file_name_poly3 = 'svm/saved_svm_kfold_poly3_scale_0_22' + str(self.i) + '.sav'
        classifier_rbf = SVC(kernel='rbf', gamma=0.22)
        file_name_rbf = 'svm/saved_svm_kfold_scale_0_22' + str(self.i) + '.sav'
        # print(input_data.shape,train_labels.shape)

        if not os.path.isfile(file_name_poly2):
            classifier_poly2.fit(self.train_input, self.train_label)
            joblib.dump(classifier_poly2, file_name_poly2)

        else:
            classifier_poly2 = joblib.load(file_name_poly2)
        ####################################
        if not os.path.isfile(file_name_poly3):
            classifier_poly3.fit(self.train_input, self.train_label)
            joblib.dump(classifier_poly3, file_name_poly3)

        else:
            classifier_poly3 = joblib.load(file_name_poly3)
        #####################################
        if not os.path.isfile(file_name_linear):
            classifier_linear.fit(self.train_input, self.train_label)
            joblib.dump(classifier_linear, file_name_linear)

        else:
            classifier_linear = joblib.load(file_name_linear)
        ######################################
        if not os.path.isfile(file_name_rbf):
            classifier_rbf.fit(self.train_input, self.train_label)
            joblib.dump(classifier_rbf, file_name_rbf)

        else:
            classifier_rbf = joblib.load(file_name_rbf)

        results_poly3 = classifier_poly3.predict(self.test_input)
        results_poly2 = classifier_poly2.predict(self.test_input)
        results_linear = classifier_linear.predict(self.test_input)
        results_rbf = classifier_rbf.predict(self.test_input)
        with open('features/svm_result_linear' + str(self.i) + '.txt', 'w') as filehandle:
            for listitem in results_linear:
                filehandle.write('%s\n' % listitem)

        with open('features/svm_result_poly2' + str(self.i) + '.txt', 'w') as filehandle:
            for listitem in results_poly2:
                filehandle.write('%s\n' % listitem)

        with open('features/svm_result_poly3' + str(self.i) + '.txt', 'w') as filehandle:
            for listitem in results_poly3:
                filehandle.write('%s\n' % listitem)

        with open('features/svm_result_rbf' + str(self.i) + '.txt', 'w') as filehandle:
            for listitem in results_rbf:
                filehandle.write('%s\n' % listitem)

        print("test numbers", len(self.test_input))
        # print("test labels : ", self.test_label)
        # print('predictions: ', results)
        print('Accuracy linear', (np.sum(results_linear == self.test_label) / len(results_linear)) * 100, '%')
        print('Accuracy poly degree2', (np.sum(results_poly2 == self.test_label) / len(results_poly2)) * 100, '%')
        print('Accuracy poly degree3', (np.sum(results_poly3 == self.test_label) / len(results_poly3)) * 100, '%')
        print('Accuracy RBF', (np.sum(results_rbf == self.test_label) / len(results_rbf)) * 100, '%')
        print("____________________________________")
