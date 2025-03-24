from flask import Flask, render_template, redirect, url_for
import subprocess
import threading

app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')  # Render the home page with buttons

import threading

def run_streamlit(script):
    subprocess.run(["streamlit", "run", script])

@app.route('/predict')
def predict():
    threading.Thread(target=run_streamlit, args=("pages/predict_page.py",), daemon=True).start()
    return redirect(url_for('home'))

# Route to the Streamlit statistics page
@app.route('/statistics')
def statistics():
    subprocess.Popen(["streamlit", "run", "pages/data_statistics.py"])  # Run Streamlit app
    return redirect(url_for('home'))  # Redirect back to home after opening

# Route to the Streamlit attrition management page
@app.route('/attrition_management')
def attrition_management():
    subprocess.Popen(["streamlit", "run", "pages/attrition_management.py"])  # Run Streamlit app
    return redirect(url_for('home'))  # Redirect back to home after opening

if __name__ == '__main__':
    app.run(debug=False)
