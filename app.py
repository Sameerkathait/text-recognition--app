import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model('mnist_model.h5')
        except:
            messagebox.showerror("Error", "Please train the model first by running train_model.py")
            root.destroy()
            return

        # Create canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)

        # Create buttons
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=20)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.RIGHT, padx=20)

        # Create prediction label
        self.prediction_label = tk.Label(root, text="Draw a digit", font=('Arial', 24))
        self.prediction_label.pack(pady=20)

        # Initialize drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 20
        self.line_color = 'black'

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.line_width, fill=self.line_color,
                                  capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a digit")

    def predict_digit(self):
        # Create image from canvas
        image = Image.new('L', (280, 280), 'white')
        draw = ImageDraw.Draw(image)
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill='black', width=self.line_width)

        # Invert image (MNIST is white digit on black background)
        image = Image.eval(image, lambda x: 255 - x)

        # Find bounding box of the digit
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        else:
            self.prediction_label.config(text="Draw a digit")
            return

        # Resize to 20x20 (MNIST digits are centered in 20x20 box)
        image = image.resize((20, 20), Image.LANCZOS)

        # Create a new 28x28 image and paste the 20x20 digit into the center
        new_image = Image.new('L', (28, 28), 'black')
        upper_left = ((28 - 20) // 2, (28 - 20) // 2)
        new_image.paste(image, upper_left)

        # Normalize
        img_array = np.array(new_image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit] * 100
        self.prediction_label.config(text=f"Prediction: {digit}\nConfidence: {confidence:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop() 