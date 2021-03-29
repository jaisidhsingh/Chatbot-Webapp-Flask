from flask import Flask, redirect, url_for, request, session
from flask import render_template
import random
import subprocess

l = ["hello", "hi", "hiya"]

app = Flask(__name__)

app.secret_key = "superMarioNintendo64"
app.config["SESSION_TYPE"] = "filesystem"

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    response = l.index(userText)  
    print(str(response))
    return str(response)

if __name__ == "__main__":    
    app.run(debug=True)
