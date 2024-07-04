from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
from flask_pymongo import PyMongo
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from passlib.hash import pbkdf2_sha256
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'

# Load face classifier and pre-trained model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\Aravinth\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

mongo = PyMongo(app)
login_manager = LoginManager(app)
login_manager.login_view = 'signin'

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def predict_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            try:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                labels.append(label)
            except Exception as e:
                print(f"Error during prediction: {e}")

    return frame, labels

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _ = predict_emotion(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})
        if existing_user and pbkdf2_sha256.verify(request.form['password'], existing_user['password']):
            user_obj = User(existing_user['_id'])
            login_user(user_obj)
            flash('Login successful!', 'success')
            return redirect(url_for('dash'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})
        if existing_user is None:
            hashed_password = pbkdf2_sha256.hash(request.form['password'])
            user_id = users.insert_one({'username': request.form['username'], 'password': hashed_password, 'email': request.form['email']}).inserted_id
            user_obj = User(user_id)
            login_user(user_obj)
            flash('Account created successfully!', 'success')
            return redirect(url_for('signin'))
        else:
            flash('Username already exists. Please choose a different one.', 'danger')
    return render_template("signup.html")

@app.route('/dash')
@login_required
def dash():
    return render_template('dash.html', username=current_user.id)

@app.route('/eman')
def eman():
    return render_template('eman.html')

@app.route('/gender')
def gender():
    return render_template('gender.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        data = {'name': name, 'email': email, 'subject': subject, 'message': message}
        try:
            mongo.db.feedback_report.insert_one(data)
            flash("Feedback Submitted Successfully", 'success')
        except Exception as e:
            flash("Error submitting form", 'danger')
        return redirect(url_for('dash'))
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
