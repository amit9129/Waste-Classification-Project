import gradio as gr
from ultralytics import YOLO            
from PIL import Image
import tempfile
import cv2
from video_view import *
import os


def image_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):
    '''
    img_path: str, path to the image
    model_id: str, model id
    image_size: int, image size
    conf_threshold: float, confidence threshold (its the minimum confidence score for a bounding box to be considered)
    iou_threshold: float, IoU threshold (its the minimum IoU score for a bounding box to be considered) [IoU = Intersection over Union]
    '''
    if model_id == 'best':
        model = YOLO(r'waste_detection\yolo\weights\best.pt')
    else:
        model = YOLO(r'waste_detection\yolo\weights\last.pt')
    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=image_size,
    )
    for r in results:
        im_array = r.plot()
        # save the image
        im = Image.fromarray(im_array[..., ::-1])
    msg = ", ".join(results[0].names.values())
    return im, "Trained to find: \n"+msg

def setup_video_inference(video, size, show_preview, save_video):
    folder = process_video(video, size, show_preview, save_video)
    video = os.listdir(folder)[0]
    path =  os.path.join(folder, video)
    print(path)
    return path
def app():
    with gr.Blocks():

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    img_path = gr.Image(type="filepath", label="Image", sources=['upload','clipboard'])
                    model_path = gr.Dropdown(
                        label="Model",
                        choices=[
                            'best',
                            'last',
                        ],
                        value="best",
                    )
                    image_size = gr.Slider(
                        label="Image Size",
                        minimum=320,
                        maximum=1280,
                        step=32,
                        value=640,
                    )
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.4,
                    )
                    iou_threshold = gr.Slider(
                        label="IoU Threshold",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                    )
                    yolo_infer = gr.Button(value="Detect waste in Image")

                with gr.Column():
                    output_numpy = gr.Image(type="numpy",label="Output")
                    output_text = gr.Label(label="Output Text")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(sources=['upload'], format="mp4")
                    frame_size = gr.Slider(label="Frame Size", minimum=320, maximum=1000, step=32, value=640)
                    show_preview = gr.Checkbox(label="Show Preview (not recommended for laptops)", value=False)
                    save_video = gr.Checkbox(label="Save Video", value=True)
                    launch_video = gr.Button(value="Detect waste in Video", )
                with gr.Column():
                    output_video = gr.Video(label="Output Video", sources=[])
                   

        yolo_infer.click(
            fn=image_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy, output_text],
        )

        launch_video.click(
            fn=setup_video_inference,
            inputs=[video, frame_size, show_preview, save_video],
            outputs=[output_video]
        )

      

gradio_app = gr.Blocks()

with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Waste Detection 
    </h1> 
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True, )