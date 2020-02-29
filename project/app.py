from flask import Flask,render_template,request,send_from_directory,jsonify
import os
import PIL
import pickle
import cv2
import numpy as np
from tensorflow.keras import models,layers,optimizers

app=Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model1=models.load_model('plant_disease.h5',compile=True)

CATEGORIES = ["Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)__Common_rust", "Corn_(maize)__healthy", "Corn(maize)__Northern_Leaf_Blight", "Potato_Early_blight", "Potato_healthy", "Potato_Late_blight", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy", "Tomato__Late_blight"]

def convert(list):
    d={}
    i=0
    for item in list:
        d[item]=i
        i+=1
    return d

l1 = ['ARHAR (TUR)', 'BAJRA', 'BANANA', 'BARLEY', 'CASTOR SEED',
       'CORIANDER', 'DRY CHILLIES', 'DRY GINGER', 'GARLIC', 'GRAM',
       'GROUNDNUT', 'HORSEGRAM', 'JOWAR', 'JUTE', 'KESARI', 'LINSEED',
       'MAIZE', 'MASOOR', 'MESTA', 'MOONG', 'ONION',
       'OTHER KHARIF PULSES', 'OTHER RABI PULSES',
       'PEAS & BEANS (PULSES)', 'POTATO', 'RAGI', 'RAPESEED &MUSTARD',
       'RICE', 'SAFFLOWER', 'SANNHAMP', 'SESAMUM', 'SMALL MILLETS',
       'SUGARCANE', 'SUNFLOWER', 'SWEET POTATO', 'TOBACCO', 'TURMERIC',
       'URAD', 'WHEAT']

l1 = convert(l1)

l2 = ['AUTUMN', 'KHARIF', 'RABI', 'SUMMER', 'TOTAL ', 'WHOLE YEAR',
       'WINTER']

l2 = convert(l2)


l3 = ['Araria', 'Arhasia', 'Arval', 'Aurangabad', 'Banka', 'Begusarai',
       'Bhagalpur', 'Bhanka', 'Bhojpur', 'Buxar', 'Darbhanga', 'Devghar',
       'Dumka', 'Gadhwa', 'Gaya', 'Giridih', 'Godda', 'Gopalganj',
       'Gumala', 'Hazaribagh', 'Jamui', 'Jehanabad', 'Kaimur (Bhabua)',
       'Katihar', 'Khagaria', 'Kishanganj', 'Lakhisarai', 'Lohardanga',
       'Madhepura', 'Madhubani', 'Munger', 'Muzaffarpur', 'Nalanda',
       'Nawada', 'Palamau', 'Pashchim Champaran', 'Patna',
       'Purba Champaran', 'Purnia', 'Ranchi', 'Rohtas', 'Saharsa',
       'Sahebganj', 'Samastipur', 'Saran', 'Sheikhpura', 'Sheohar',
       'Singhbhum', 'Singhbhur(west)', 'Sitamarhi', 'Siwan', 'Sumal',
       'Supaul', 'Vaishali', 'Zamui']

l3 = convert(l3)

app_route=os.path.dirname(os.path.abspath(__file__))

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('diag.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload',methods=['POST'])
def upload():
    target=os.path.join(app_route,'static/images/')
    if not os.path.isdir(target):
            os.mkdir(target)
    upload=request.files.getlist("file")[0]
    filename=upload.filename
    ext=os.path.splitext(filename)[1]
    if (ext==".jpg") or (ext==".png") or (ext==".bmp"):
        print("File accepted")
    else:
        return render_template("error.html",message="The selected file is not supported")
    destination="/".join([target,filename])
    upload.save(destination)
    img=cv2.imread(destination)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(150,150))
    img1=np.reshape(img,[1,150,150,3])
    img1=img1/255.0
    disease=model1.predict_classes(img1)
    prediction=disease[0]
    op=CATEGORIES[prediction]
    return render_template("dis.html",k={"image_name":filename,"text":op})

@app.route('/disease')
def disease():
    return render_template('dis.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output=round(prediction[0],3)
    return render_template('prediction.html',prediction='Yield should be {}'.format(output))

# @app.route('/results',methods=['POST','GET'])
# def results():
#     data = request.get_json(force=True)
#     print(data)
#     data['SEASON'] = float(l2[data['SEASON']])
#     data['DISTRICT'] = float(l3[data['DISTRICT']])
#     data['AREA']=float(14[data['AREA']])
#     data['CROP'] = float(l1[data['CROP']])
#     prediction = model.predict([np.array(list(data.values()))])
#     output = round(prediction[0],2)
#     return jsonify(output)



if __name__=="__main__":
    app.run(debug=True)