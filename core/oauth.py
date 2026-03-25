import os
import requests
from flask import Flask, request, jsonify, redirect
from flask_oauthlib.client import OAuth

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')

# Configuring OAuth for Google and GitHub
oauth = OAuth(app)

google = oauth.remote_app(
    'google',
    consumer_key=os.environ.get('GOOGLE_CLIENT_ID'),
    consumer_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    request_token_params={'scope': 'email'},
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

github = oauth.remote_app(
    'github',
    consumer_key=os.environ.get('GITHUB_CLIENT_ID'),
    consumer_secret=os.environ.get('GITHUB_CLIENT_SECRET'),
    base_url='https://api.github.com/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
)

@app.route('/login/google')
def login_google():
    return google.authorize(callback="http://127.0.0.1:5000/login/google/authorized")

@app.route('/login/google/authorized')
def google_authorized():
    response = google.get('userinfo')
    return jsonify(response.data)

@app.route('/login/github')
def login_github():
    return github.authorize(callback="http://127.0.0.1:5000/login/github/authorized")

@app.route('/login/github/authorized')
def github_authorized():
    response = github.get('/user')
    return jsonify(response.data)

if __name__ == '__main__':
    app.run(debug=True)