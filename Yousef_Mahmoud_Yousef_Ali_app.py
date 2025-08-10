import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import warnings as wr
import traceback
from flask import Flask
import os
wr.filterwarnings('ignore')


app = Flask(__name__)




try:
    model = pickle.load(open("Model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    label_encoder = pickle.load(open("label_encoder.pkl","rb"))
    model_features = pickle.load(open("model_features.pkl","rb"))
    print("All files loaded")
except FileNotFoundError as e:
    print(f"Error loading the file: {e}")
    exit()
except Exception as e:
    print(f"Error occured while loading the file: {e}")
    exit()

@app.route("/")
def Home():
    return render_template("Yousef_Mahmoud_Yousef_Ali_index.html")

def preprocessing_single_inputs(input_data):
    data_file = pd.DataFrame([input_data])

    data_file.columns = [col.strip() for col in data_file.columns]

    col_to_drop = ['repeated', 'P-not-C', 'car parking space', 'number of week nights', 'Booking_ID']
    data_file = data_file.drop(columns=[col for col in col_to_drop if col in data_file.columns], errors='ignore')


    if 'date of reservation' in data_file.columns and pd.notna(data_file['date of reservation'].iloc[0]):
        data_file['date of reservation'] = pd.to_datetime(data_file['date of reservation'], errors='coerce')
        data_file['reservation_day'] = data_file['date of reservation'].dt.day
        data_file['reservation_weekday'] = data_file['date of reservation'].dt.dayofweek
        data_file = data_file.drop(columns=['date of reservation'], errors='ignore')
    else:
        data_file['reservation_day'] = 0
        data_file['reservation_weekday'] = 0 #If the dates are missing or invalid in input, fill with a default value

    cate_for_oh_encoding =  ['type of meal', 'room type', 'market segment type']
    for col in cate_for_oh_encoding:
        if col in data_file.columns:
            data_file = pd.get_dummies(data_file, columns=[col], drop_first=True, prefix=col)
            # If a categorical column is not in input, the dummy columns will be added as 0's below -->
    

    missing_col = set(model_features) - set(data_file.columns)
    for co in missing_col:
        data_file[co] = 0
    
    extra_col = set(data_file.columns) - set(model_features)
    data_file = data_file.drop(columns=list(extra_col), errors='ignore')

    data_file = data_file[model_features]

    data_file_scaler = scaler.transform(data_file)

    return data_file_scaler


@app.route("/predict", methods = ["POST"])
def Predict():
    try:
        input_data = request.form.to_dict()

        required_form_fields = [
            'lead time', 'average price', 'special requests',
            'number of adults', 'number of children', 'number of weekend nights',
            'number of previous bookings', 'date of reservation',
            'type of meal', 'room type', 'market segment type'
        ]
        for f in required_form_fields:
            if f not in input_data or not input_data[f].strip():
                return render_template("Yousef_Mahmoud_Yousef_Ali_index.html", predict_texts=f"Error, please fill all the fields. MissingL: {f}")



        numeric_features = ['lead time', 'average price', 'special requests', 'number of adults',
                              'number of children', 'number of weekend nights', 'number of week nights',
                              'number of previous bookings']
        for k in numeric_features:
            if k in input_data:
                input_data[k] = float(input_data[k])

        processed_features = preprocessing_single_inputs(input_data)

        predict_numeirc = model.predict(processed_features)
        predict_texts = label_encoder.inverse_transform(predict_numeirc)

        return render_template("Yousef_Mahmoud_Yousef_Ali_index.html", prediction_text= f"Hotel Booking prediction: {predict_texts[0]}")

    except Exception as e:
        print(f"Error while making the prediction: {e}")
        traceback.print_exc();
        return render_template("Yousef_Mahmoud_Yousef_Ali_index.html", predict_texts=f"error in prediction: {e}")
    
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

app = Flask(__name__)

def home():
    return "Hello, this is the homepage!"
