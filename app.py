import matplotlib

matplotlib.use("Agg")

import os
from flask import Flask, render_template, request
import numpy as np
import cv2
from model import dummy
from dicompylercore import dicomparser
import time
from logging import FileHandler, WARNING
import json

# clear_directory = "static"
# list(map(os.unlink, (os.path.join(clear_directory, file) for file in os.listdir(clear_directory))))

start = time.time()

application = Flask(__name__)
file_handler = FileHandler("log.txt")
file_handler.setLevel(WARNING)
application.logger.addHandler(file_handler)
print("Initialization time: {0} seconds.".format(int(time.time() - start)))

# Thresholds for pathologies.
thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


@application.after_request
def header(_request):
    _request.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    _request.headers["Pragma"] = "no-cache"
    _request.headers["Expires"] = "0"
    _request.headers["Cache-Control"] = "public, max-age=0"

    return _request


@application.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@application.route("/process", methods=["POST"])
def process():
    original_stream = request.files["load"]
    filename = os.path.splitext(original_stream.filename)[0]

    if os.path.splitext(original_stream.filename)[1] == ".dcm":
        original_stream.save("static/{0}.dcm".format(filename))
        parsed = dicomparser.DicomParser("static/{0}.dcm".format(filename))
        image = np.array(parsed.GetImage(), dtype=np.uint8)

        if parsed.GetImageData()["photometricinterpretation"] == "MONOCHROME1":
            image = 255 - image

        cv2.imwrite("static/{0}.png".format(filename), image)
        os.remove("static/{0}.dcm".format(filename))
    else:
        original_stream.save("static/{0}.png".format(filename))

    pathology_probability, diagnosed_diseases = dummy.process("static/{0}.png".format(filename),
                                                              "static/{0}_cam.png".format(filename))
    logical = pathology_probability > thresholds
    status = "Probability: " + str(pathology_probability[logical]) + "%. Diseases: " + str(
        diagnosed_diseases[logical]) + "."

    if len(diagnosed_diseases[logical]) > 0:
        messages = {"original_image": "static/{0}.png".format(filename),
                    "processed_image": "static/{0}_cam.png".format(filename),
                    "status": status}
    else:
        messages = {"original_image": "static/{0}.png".format(filename),
                    "processed_image": "",
                    "status": status}

    return render_template("result.html", messages=messages)


# Example of request: curl -F "=@/path/to/file/12345.jpg" http://127.0.0.1:5000/api
@application.route("/api", methods=["POST"])
def api():
    original_stream = request.files['file']
    filename = os.path.splitext(original_stream.filename)[0]

    if os.path.splitext(original_stream.filename)[1] == ".dcm":
        original_stream.save("static/{0}_api.dcm".format(filename))
        parsed = dicomparser.DicomParser("static/{0}_api.dcm".format(filename))
        image = np.array(parsed.GetImage(), dtype=np.uint8)

        if parsed.GetImageData()["photometricinterpretation"] == "MONOCHROME1":
            image = 255 - image

        cv2.imwrite("static/{0}_api.png".format(filename), image)
        os.remove("static/{0}_api.dcm".format(filename))
    else:
        original_stream.save("static/{0}_api.png".format(filename))

    pathology_probability, diagnosed_diseases = dummy.process("static/{0}_api.png".format(filename))
    logical = pathology_probability > thresholds

    return json.dumps({"Labels": list(diagnosed_diseases[logical]), "id": original_stream.filename})


if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=True)
