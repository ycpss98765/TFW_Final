import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pywt


def readinput(train_data_number, number_per_person):
    person_list = ["Aaron_Peirsol", "Abdoulaye_Wade", "Abdullah"]
    train_path_list = []
    train_path= ""
    for person in person_list:    
        for number in range(1, number_per_person + 1):
            train_path = "database/train/" + person + "_000" + str(number) + ".pgm"
            train_path_list.append(train_path)

    train_data = np.zeros((len(train_path_list), 64, 64))
    for i in range(len(train_path_list)):
        train_data[i]=(cv.imread(train_path_list[i], cv.IMREAD_GRAYSCALE))


    train_label = np.array([0, 0, 1, 1, 2, 2])
    return train_data, train_label




def DWT(train_data, transform_num = 8):
    #train_data_transformed = []
    LL_list = []
    for i in range(len(train_data)):
        LL = train_data[i]
        for num in range(transform_num):
            coeffs2 = pywt.dwt2(LL, 'bior1.3')
            LL, (LH, HL, HH) = coeffs2
        LL_list.append(LL)
    train_data_LL = np.asarray(LL_list)
    return train_data_LL

def convert2vector(train_data_LL):
    train_data_vector = []
    for i in range(len(train_data_LL)):
        LL = train_data_LL[i]
        LL_1d_vector = LL.flatten()
        train_data_vector.append(LL_1d_vector)
    
    train_data_vector = np.asarray(train_data_vector)
    return train_data_vector

def LDA(train_data_vector, train_label):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data_vector, train_label)
    return clf

def predict(model, test_data): 
    predict_label = model.predict(test_data)
    predict_prob = model.predict_proba(test_data)
    return predict_label, predict_prob




train_data_number = 3 # fixed, can not be modified
number_per_person = 2 # fixed, can not be modified

train_data, train_label = readinput(train_data_number=3, number_per_person=2)
"""
Here is the result without DWT transfrom
"""

print("-----Using WDT transform-----")
train_data_vector = convert2vector(train_data)
model = LDA(train_data_vector, train_label)
predict_label, predict_prob = predict(model, train_data_vector)
Accuracy = sum(predict_label == train_label) / len(train_label)
print("The predicted label are: \n", predict_label)
print("The predicted probability for each label: \n", predict_prob)
print("Accuracy: \n", Accuracy)



"""
Here is the result with DWT transfrom
"""

print("\n-----Not using WDT transform-----")
train_data_LL = DWT(train_data, transform_num = 4)
train_data_vector = convert2vector(train_data_LL)
model = LDA(train_data_vector, train_label)
predict_label, predict_prob = predict(model, train_data_vector)
Accuracy = sum(predict_label == train_label) / len(train_label)
print("The predicted label are: \n", predict_label)
print("The predicted probability for each label: \n", predict_prob)
print("Accuracy: \n", Accuracy)





from sklearn.decomposition import PCA

#pca = PCA(n_components=2)
#pca.fit(train_data_vector)
#x = pca.transform(train_data_vector)

clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(train_data_vector, train_label)
x = clf.transform(train_data_vector)

plt.title("Waveletfaces with LDA to 2D-features")
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.scatter(x[:2, 0], x[:2, 1])
plt.scatter(x[2:4, 0], x[2:4, 1])
plt.scatter(x[4:6, 0], x[4:6, 1])
