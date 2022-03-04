import pickle
import numpy as np
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
features = np.array([[2005, 12000, 1, 2,4,17.0,1400.0,5.0]])
result = loaded_model.predict(features)
print(result)