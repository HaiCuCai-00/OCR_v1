import sys

import cv2
import gradio as gr

sys.path.append("..")
from tools import Pipeline

pipeline = Pipeline()


def inference(image, type):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pipeline.start(image, type)


iface = gr.Interface(
    inference,
    inputs=[
        gr.inputs.Image(type="numpy", tool="editor", source="upload"),
        gr.inputs.Radio(["cmnd","cccd","cccc", "abt", "ttk", "hc", "dkkd", "sohong"]),
    ],
    outputs="json",
    verbose=True,
    server_name="0.0.0.0"
)

iface.test_launch()

if __name__ == "__main__":
    iface.launch(debug=True)
