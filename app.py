import gradio as gr

from transformers import pipeline

pipe = pipeline(task="text-classification", 
                 model="finiteautomata/bertweet-base-sentiment-analysis")


gr.Interface.from_pipeline(pipe, 
                           title="Bertweet Base Sentiment Analysis",
                           description="Tweet Classification using POS, NEU, NEG labels",
                           allow_flagging="never").launch(inbrowser=True)