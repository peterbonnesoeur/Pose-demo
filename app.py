from PIL import Image
import torch
import gradio as gr
import openpifpaf
import numpy as np

predictor_pose = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
predictor_whole_body = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
predictor_vehicle = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-24')


def inference(img, ver):
    if ver == 'pose':
        predictor = predictor_pose
    elif ver == 'whole-body':
        predictor = predictor_whole_body
    elif ver == 'vehicles':
        predictor = predictor_vehicle
    else:
        raise ValueError('invalid version')

    predictions, gt_anns, image_meta = predictor.pil_image(img)
    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.image_canvas(img, fig_file = "test.jpg") as ax:
        annotation_painter.annotations(ax, predictions)

    out = Image.open("test.jpg")
    return out
      
  
title = "Openpifpaf"
description = "Gradio demo for openpifpaf. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below and don't hesitate to SMASH THAT LIKE BUTTON (wait, wrong platform)"
article = "<p style='text-align: center'><a href='https://github.com/openpifpaf/openpifpaf' target='_blank'>Github Repo Openpifpaf</a> | <a href='https://github.com/peterbonnesoeur' target='_blank'>Github Repo peterbonnesoeur</a></p>"

with open("article.html", "r", encoding='utf-8') as f:
    article= f.read()

examples=[ 
    ['basketball.jpg','whole-body'],
    ['meeting.jpeg','whole-body'],
    ['crowd.jpg','pose'],
    ['elon.png','whole-body'],
    ['billie.png','whole-body'],
    ['india-vehicles.jpeg', 'vehicles'],
    ['russia-vehicles.jpg', 'vehicles'],
    ['paris-vehicles.jpg', 'vehicles'],
    ]
gr.Interface(inference, [gr.inputs.Image(type="pil"),gr.inputs.Radio(['pose', 'whole-body', 'vehicles'], type="value", default='pose', label='version')
], gr.outputs.Image(type="pil"),title=title,description=description,article=article,enable_queue=True,examples=examples).launch()