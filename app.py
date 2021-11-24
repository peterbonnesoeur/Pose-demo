import os
import gradio as gr

def inference(img, ver, white_overlay):

    if white_overlay:
        white_overlay = "--white-overlay=0.3"
    else:
        white_overlay = ""

    if ver == 'pose':
        os.system("python -m openpifpaf.predict "+img.name+" --checkpoint=shufflenetv2k30 --line-width=4 " + white_overlay + " -o out.jpg")
    elif ver == 'whole-body':
        os.system("python -m openpifpaf.predict "+img.name+" --checkpoint=shufflenetv2k30-wholebody --instance-threshold 0.05 " + white_overlay + " --seed-threshold 0.05 \
                                                             --line-width 3 -o out.jpg")
    elif ver == 'vehicles':
        os.system("python -m openpifpaf.predict "+img.name+" --checkpoint=shufflenetv2k16-apollo-24 --line-width=5  " + white_overlay + " -o out.jpg")
    elif ver == 'animal':
        os.system("python -m openpifpaf.predict "+img.name+" --checkpoint=shufflenetv2k30-animalpose --line-width=5 --font-size=6 " + white_overlay + " \
                   --long-edge=500  -o out.jpg")
    else:
        raise ValueError('invalid version')

    return "out.jpg"
      
  
title = "Openpifpaf - pose estimation for human, vehicles and animals"
description = "Gradio demo for openpifpaf. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below and don't hesitate to SMASH THAT LIKE BUTTON (and you do not have a dislike there either so...)"
article = "<p style='text-align: center'><a href='https://github.com/openpifpaf/openpifpaf' target='_blank'>Github Repo Openpifpaf</a> | <a href='https://github.com/peterbonnesoeur' target='_blank'>Github Repo peterbonnesoeur</a></p>"

with open("article.html", "r", encoding='utf-8') as f:
    article= f.read()

examples=[ 
    ['basketball.jpg','whole-body'],
    ['bill.png','whole-body'],
    ['billie.png','whole-body'],
    ['meeting.jpeg','pose'],
    ['crowd.jpg','pose'],
    ['dalmatian.jpg', 'animal'],
    ['tappo_loomo.jpg', 'animal'],
    ['cow.jpg', 'animal'],
    ['india-vehicles.jpeg', 'vehicles'],
    ['russia-vehicles.jpg', 'vehicles'],
    ['paris-vehicles.jpg', 'vehicles'],

    ]

gr.Interface(
    inference,
    [
        gr.inputs.Image(type="file", label="Input"),
        gr.inputs.Radio(['whole-body', 'pose', 'vehicles', 'animal'], type="value", default='whole-body', label='version'),
        gr.inputs.Checkbox(default=False, label="White overlay")
    ],
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    enable_queue=True,
    examples=examples).launch()