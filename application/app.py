# Digit Recognizer GUI
# Author: Insoo Son (insooeric.son@gmail.com)
# Date: April 28th, 2025

########## IMPORTANT ##########
# if the drawn image is too small, 
# prediction may be wrong

import os
import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
from PIL import Image, ImageDraw

########## CONSTANTS ##########
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CANVAS_SIZE = 280
IMG_SIZE = 28
THICK_PASSES = 7
BG_COLOR = "#2e2e2e"
FONT_PREDICT = ("Arial", 14, "bold")
FONT_CONFIDENCE = ("Arial", 12)
DRAW_TAG = "drawn"

########## DEFINE NEURAL NETWORK ##########
class NeuralNetwork:
    def __init__(self, input_layer, hidden1, hidden2, hidden3, hidden4, output_layer):
        # weights and biases will be loaded from an HDF5 model
        self.weight1 = None; self.bias1   = None
        self.weight2 = None; self.bias2   = None
        self.weight3 = None; self.bias3   = None
        self.weight4 = None; self.bias4   = None
        self.weight5 = None; self.bias5   = None

    def forward_propagation(self, input_layer):
        # five-layer MLP with ReLU activations and softmax output
        # FYI: this is explained in training_model.ipynb
        hidden_layer_1 = np.maximum(0, input_layer.dot(self.weight1) + self.bias1)
        hidden_layer_2 = np.maximum(0, hidden_layer_1.dot(self.weight2) + self.bias2)
        hidden_layer_3 = np.maximum(0, hidden_layer_2.dot(self.weight3) + self.bias3)
        hidden_layer_4 = np.maximum(0, hidden_layer_3.dot(self.weight4) + self.bias4)
        output_layer = hidden_layer_4.dot(self.weight5) + self.bias5
        shift = output_layer - np.max(output_layer, axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities

########## UTILITIES ##########
# dilate non-zero pixels in all directions for a given number of passes
def thicken(arr: np.ndarray, passes) -> np.ndarray:
    for _ in range(passes):
        up    = np.vstack((arr[1:, :], np.zeros((1, arr.shape[1]), dtype=arr.dtype)))
        down  = np.vstack((np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr[:-1, :]))
        left  = np.hstack((arr[:, 1:], np.zeros((arr.shape[0], 1), dtype=arr.dtype)))
        right = np.hstack((np.zeros((arr.shape[0], 1), dtype=arr.dtype), arr[:, :-1]))
        arr = np.maximum.reduce([arr, up, down, left, right])
    return arr

# instantiate a NeuralNetwork and load weights from the given HDF5 file
def load_model(path: str) -> NeuralNetwork:
    model = NeuralNetwork(IMG_SIZE*IMG_SIZE, 512, 256, 128, 64, 10)
    with h5py.File(path, "r") as hf:
        model.weight1, model.bias1 = hf["weight1"][:], hf["bias1"][:]
        model.weight2, model.bias2 = hf["weight2"][:], hf["bias2"][:]
        model.weight3, model.bias3 = hf["weight3"][:], hf["bias3"][:]
        model.weight4, model.bias4 = hf["weight4"][:], hf["bias4"][:]
        model.weight5, model.bias5 = hf["weight5"][:], hf["bias5"][:]
    return model

########## CONSTRUCT GUI ##########
def build_gui():
    root = tk.Tk()
    root.title("Digit Recognizer")
    root.geometry("300x350")
    root.resizable(False, False)
    root.configure(bg=BG_COLOR)

    ########## top frame ##########
    top_frame = tk.Frame(root, height=50, bg=BG_COLOR)
    top_frame.pack(fill="x"); top_frame.pack_propagate(False)

    # labels container
    labels_frame = tk.Frame(top_frame, bg=BG_COLOR)
    labels_frame.pack(side="left", padx=10)

    # prediction label
    predict_label = tk.Label(
        labels_frame,
        text="Predict: N/A",
        font=FONT_PREDICT,
        bg=BG_COLOR,
        fg="white"
    )
    predict_label.pack(anchor="w")

    # confidency label
    confidence_label = tk.Label(
        labels_frame,
        text="Confidence: N/A",
        font=FONT_CONFIDENCE,
        bg=BG_COLOR,
        fg="white"
    )
    confidence_label.pack(anchor="w")

    # import button
    import_btn = tk.Button(top_frame, text="Import Model")
    import_btn.pack(side="right", padx=10)

    ########## bottom frame ##########
    bottom_frame = tk.Frame(root, bg=BG_COLOR)
    canvas = tk.Canvas(
        bottom_frame,
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        bg="white",
        highlightthickness=0
    )
    pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(pil_img)

    ########## DRAWING STATE ##########
    last_pos = {"x": None, "y": None}

    ########## DRAWING UTILITIES ##########
    # clear canvas
    def clear_canvas():
        canvas.delete(DRAW_TAG)
        draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        predict_label.config(text="Predict: N/A")
        confidence_label.config(text="Confidence: N/A")

    # on start drawing
    def start_draw(event):
        last_pos["x"], last_pos["y"] = event.x, event.y
        r = 5
        canvas.create_oval(
            event.x-r, event.y-r, 
            event.x+r, event.y+r,
            fill="black", outline="black",
            tags=DRAW_TAG
        )
        draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill=0)

    # on drawing
    def draw_motion(event):
        x, y = event.x, event.y
        if last_pos["x"] is not None:
            canvas.create_line(
                last_pos["x"], 
                last_pos["y"], 
                x, y,
                fill="black", 
                width=10, 
                capstyle="round",
                tags=DRAW_TAG
            )
            draw.line([last_pos["x"], last_pos["y"], x, y], fill=0, width=10)
        last_pos["x"], last_pos["y"] = x, y

    # trigger prediction
    def predict(event):
        arr = np.array(pil_img, dtype=np.uint8)

        # if over 50% of the pixels are white, it means the background is white
        # and the number is drawn in black
        # so, we invert it by subtracting all pixels
        if np.mean(arr > 254) > 0.5:
            arr = 255 - arr

        # then we thicken the original image by 7 (THICK_PASSES)
        arr = thicken(arr, THICK_PASSES)

        # then resize image to 28 x 28 (IMG_SIZE x IMG_SIZE)
        img28 = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE))

        # at this point, it is guarenteed that
        #   - black cells represents background
        #   - white and gray cells represents digit

        # there could be a case where AI won't properly evaluate when the digit is skewed into one side
        # for example, when user draws 1 at the top right corner of the canvas
        # AI may predict it as 3 since it reads the entire canvas then parses it
        # what we will do is to crop only digit cells, then refill top, bottom, left, right so that the digit is vertically and horizontally centered
        
        # to solve this problem, we will:
        #   1. scan top to bottom; check if any row contains non-zero (digit cell). if so, remove every row above
        #   2. scan bottom to top; check if any row contains non-zero (digit cell). if so, remove every row below
        #   3. scan left to right; check if any column contains non-zero (digit cell). if so, remove every column behind
        #   4. scan right to left; check if any column contains non-zero (digit cell). if so, remove every column after
        # this way, we'll have 2d array where the digits are fitted (shape is not 28 x 28)
        arr28 = np.asarray(img28, dtype=np.uint8)
        mask = arr28 > 0
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)

        if rows.any():
            top = np.argmax(rows)
            bottom = len(rows) - 1 - np.argmax(rows[::-1])
        else:
            top, bottom = 0, IMG_SIZE-1

        if cols.any():
            left = np.argmax(cols)
            right = len(cols) - 1 - np.argmax(cols[::-1])
        else:
            left, right = 0, IMG_SIZE-1

        crop = arr28[top:bottom+1, left:right+1]

        # now that we have cropped 2d array, we need to refill the array so that the shape is 28 x 28
        # we will grab that array, then fill top and bottom recursively until the overall number of rows are 28
        # then, we will grab that once again to fill left and top recursively until the overall number of columns are 28
        # this way, we'll have 2d array with 28 x 28 where the digit is centered horizontally and vertically
        h, w = crop.shape
        centered_2d_img = crop


        for i in range(IMG_SIZE - h):
            row = np.zeros((1, w), dtype=np.uint8)
            centered_2d_img = np.vstack((row, centered_2d_img)) if i % 2 == 0 else np.vstack((centered_2d_img, row))
        for j in range(IMG_SIZE - w):
            col = np.zeros((IMG_SIZE, 1), dtype=np.uint8)
            centered_2d_img = np.hstack((col, centered_2d_img)) if j % 2 == 0 else np.hstack((centered_2d_img, col))

        # just in case you would like to see the processed image
        # below code will save the processed image as png in current directory as "processed_img.png"
        # you may uncomment this out
        # Image.fromarray(centered_2d_img).save(os.path.join(CURRENT_DIR, "processed_img.png"))

        # now that we have centered image in 2d array, we'll flatten then normalize it
        # so that we can forward propagate for evaluation
        flat = centered_2d_img.flatten().astype(np.float32) / 255.0
        probs = model.forward_propagation(flat[None, :])
        pred = int(np.argmax(probs))
        conf = float(np.max(probs)) * 100
        predict_label.config(text=f"Predict: {pred}")
        confidence_label.config(text=f"Confidence: {conf:.1f}%")

    # import HDF5 model
    def import_and_activate():
        global model # make this global since we're using it in other functions
        # clicking this button will allow user to select HDF5 model (.h5)
        path = filedialog.askopenfilename(
            title="Select Model File", filetypes=[("H5 files", "*.h5")]
        )
        if not path:
            return
        
        # load and enable drawing
        model = load_model(path)
        clear_btn = tk.Button(canvas, text="Clear", command=clear_canvas)
        canvas.create_window(10, 10, anchor="nw", window=clear_btn)
        canvas.pack(expand=True)
        bottom_frame.pack(fill="both", expand=True)
        canvas.bind("<Button-1>", start_draw)
        canvas.bind("<B1-Motion>", draw_motion)
        canvas.bind("<ButtonRelease-1>", predict)

    import_btn.config(command=import_and_activate)
    root.mainloop()

########## MAIN ##########
if __name__ == "__main__":
    build_gui()