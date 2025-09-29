"""Download a dataset from Roboflow.
"""

from roboflow import Roboflow

rf = Roboflow(api_key="6DaO7Ncq2dJDFDWfIqup")
project = rf.workspace("cnnonboard").project("boat-detection-model-2")
version = project.version(1)
dataset = version.download("yolov11")
                