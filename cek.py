import pickle

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model('RandomForest_Productivity_3.pkl')


# Memastikan model adalah RandomForest
if hasattr(model, 'feature_importances_'):
    # Mendapatkan pentingnya fitur
    feature_importances = model.feature_importances_
    print("Feature importances:", feature_importances)
else:
    print("Model tidak memiliki atribut 'feature_importances_'")
