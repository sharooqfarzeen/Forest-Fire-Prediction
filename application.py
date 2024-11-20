from flask import Flask, render_template, request
import pickle

application = Flask(__name__)
app = application

# Import Standard Scaler and Ridge Regression models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Function which determines fire danger based on result returned from model
def find_danger(fwi: float) -> str:
    """
    The fire weather index can be categorised into 6 classes of danger as follows: 
    Very low danger: FWI is less than 5.2. 
    Low danger: FWI is between 5.2 and 11.2. 
    Moderate danger: FWI is between 11.2 and 21.3. 
    High danger: FWI is between 21.3 and 38.0.
    """

    danger = ""

    if fwi < 5.2:
        danger = "Very low danger of fire."
    elif fwi < 11.2:
        danger = "Low danger of fire."
    elif fwi < 21.2:
        danger = "Moderate danger of fire."
    else:
        danger = "High danger of fire."

    return danger


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit():
    Temperature=float(request.form.get('Temperature'))
    RH = float(request.form.get('RH'))
    Ws = float(request.form.get('Ws'))
    Rain = float(request.form.get('Rain'))
    FFMC = float(request.form.get('FFMC'))
    ISI = float(request.form.get('ISI'))
    BUI = float(request.form.get('BUI'))
    Region = float(request.form.get('Region'))

    new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,ISI,BUI,Region]])
    fwi = round(ridge_model.predict(new_data_scaled)[0], 2)

    danger = find_danger(fwi)

    return render_template('result.html', fwi=fwi, danger=danger)  

if __name__=="__main__":
    app.run(host="0.0.0.0") 