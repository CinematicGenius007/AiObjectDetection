from flask import Flask, render_template, request, redirect
from threading import Thread
from werkzeug.utils import secure_filename
import os
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import process_file as pf


UPLOAD_FOLDER = 'static/'

app = Flask('app')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == "POST":
        if 'media_file' not in request.files:
            return redirect('/')
        media = request.files['media_file']
        if media.filename == '':
            return redirect('/')
        # media.save(secure_filename(media.filename))
        media_file = secure_filename(media.filename)
        # file_loc = os.path.join(app.config['UPLOAD_FOLDER'], media_file)
        media.save(os.path.join(app.config['UPLOAD_FOLDER'], media_file))

        if 'mp4' in media_file:
            new_file_path = pf.show(os.path.join(app.config['UPLOAD_FOLDER'], media_file))
            return render_template('video_final.html', url=new_file_path)
        elif media_file.split('/')[-1].split('.')[-1] in ['jpg', 'jpeg', 'png']:
            new_file_path = pf.show(os.path.join(app.config['UPLOAD_FOLDER'], media_file))
        
            return render_template('final_product.html', url=new_file_path)
    elif request.method == "GET":
        return render_template('index.html')

def run():
    app.run(host='0.0.0.0', port=8098)


def keep_alive():
    t = Thread(target=run)
    t.start()

if __name__ == '__main__':
    keep_alive()