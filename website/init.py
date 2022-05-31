from flask import Flask, render_template, request, redirect                 # RETURN OPTIONS.
from werkzeug.utils import secure_filename
import os
from flask import flash, request, url_for, send_file, Markup                # URL AND OPTIONS.
from PIL import Image, ImageDraw
import sys
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

######################## FLASK APP NAME AND SECRET KEY.
app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def VisionTransformer():    
    os.system("python inference.py --thresh 0.9 --resume /u/markmartori/detr/output_COCO/checkpoint.pth")
    return True



####################### APPLICATION ROUTES!
@app.route("/")
def home():
    return render_template("home.html") # , detected_image = img)

@app.route('/upload_file', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'input_img' not in request.files:
            print("not img found")
            return redirect(request.url)
        file = request.files['input_img']
        img = Image.open(file.stream)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            flash('File found')
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img.save("/u/markmartori/website/static/images/selected/a.png")
            detected_img = VisionTransformer()
            if detected_img:
                return render_template('results.html')
            else:
                return render_template('error.html')
    return redirect(request.url)                                

if __name__ == "__main__":
    app.run(debug=True)
  

