import numpy as np
import sys
base_path = sys.path[0]
new_added_person_name = 'ly' 
f = open(base_path + '/' + new_added_person_name + "/new_feature.txt")
line = f.readline()
features = []
while line:
    feature = line.split(',')
    feature.pop()
    features.append(list(map(float, feature)))
    line = f.readline()
f.close()
f = open(base_path + '/' + 'feature_average.txt', 'a+')
feature_numpy = np.array(features)
feature_average = np.average(feature_numpy, axis=0)
feature_string = new_added_person_name + ','
for i in range(len(feature_average)):
    feature_string += str(feature_average[i])
    feature_string += ","
feature_string += '\n'
f.writelines(feature_string)
f.close()