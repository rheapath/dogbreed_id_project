import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tkinter.ttk import Frame, Button, Style, Label, Progressbar
from PIL import Image, ImageTk
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import keras.backend

SAVED_MODEL_PATH = './saved_model'
SAVED_MODEL_UNTRAINED_PATH = './saved_model_untrained' # use ./saved_model if untrained not available
WIDTH = 1024
HEIGHT = 512

LABEL_NAMES = {0: 'affenpinscher', 1: 'afghan_hound', 2: 'african_hunting_dog', 3: 'airedale', 4: 'american_staffordshire_terrier', 5: 'appenzeller', 6: 'australian_terrier', 7: 'basenji', 8: 'basset', 9: 'beagle', 10: 'bedlington_terrier', 11: 'bernese_mountain_dog', 12: 'black-and-tan_coonhound', 13: 'blenheim_spaniel', 14: 'bloodhound', 15: 'bluetick', 16: 'border_collie', 17: 'border_terrier', 18: 'borzoi', 19: 'boston_bull', 20: 'bouvier_des_flandres', 21: 'boxer', 22: 'brabancon_griffon', 23: 'briard', 24: 'brittany_spaniel', 25: 'bull_mastiff', 26: 'cairn', 27: 'cardigan', 28: 'chesapeake_bay_retriever', 29: 'chihuahua', 30: 'chow', 31: 'clumber', 32: 'cocker_spaniel', 33: 'collie', 34: 'curly-coated_retriever', 35: 'dandie_dinmont', 36: 'dhole', 37: 'dingo', 38: 'doberman', 39: 'english_foxhound', 40: 'english_setter', 41: 'english_springer', 42: 'entlebucher', 43: 'eskimo_dog', 44: 'flat-coated_retriever', 45: 'french_bulldog', 46: 'german_shepherd', 47: 'german_short-haired_pointer', 48: 'giant_schnauzer', 49: 'golden_retriever', 50: 'gordon_setter', 51: 'great_dane', 52: 'great_pyrenees', 53: 'greater_swiss_mountain_dog', 54: 'groenendael', 55: 'ibizan_hound', 56: 'irish_setter', 57: 'irish_terrier', 58: 'irish_water_spaniel', 59: 'irish_wolfhound', 60: 'italian_greyhound', 61: 'japanese_spaniel', 62: 'keeshond', 63: 'kelpie', 64: 'kerry_blue_terrier', 65: 'komondor', 66: 'kuvasz', 67: 'labrador_retriever', 68: 'lakeland_terrier', 69: 'leonberg', 70: 'lhasa', 71: 'malamute', 72: 'malinois', 73: 'maltese_dog', 74: 'mexican_hairless', 75: 'miniature_pinscher', 76: 'miniature_poodle', 77: 'miniature_schnauzer', 78: 'newfoundland', 79: 'norfolk_terrier', 80: 'norwegian_elkhound', 81: 'norwich_terrier', 82: 'old_english_sheepdog', 83: 'otterhound', 84: 'papillon', 85: 'pekinese', 86: 'pembroke', 87: 'pomeranian', 88: 'pug', 89: 'redbone', 90: 'rhodesian_ridgeback', 91: 'rottweiler', 92: 'saint_bernard', 93: 'saluki', 94: 'samoyed', 95: 'schipperke', 96: 'scotch_terrier', 97: 'scottish_deerhound', 98: 'sealyham_terrier', 99: 'shetland_sheepdog', 100: 'shih-tzu', 101: 'siberian_husky', 102: 'silky_terrier', 103: 'soft-coated_wheaten_terrier', 104: 'staffordshire_bullterrier', 105: 'standard_poodle', 106: 'standard_schnauzer', 107: 'sussex_spaniel', 108: 'tibetan_mastiff', 109: 'tibetan_terrier', 110: 'toy_poodle', 111: 'toy_terrier', 112: 'vizsla', 113: 'walker_hound', 114: 'weimaraner', 115: 'welsh_springer_spaniel', 116: 'west_highland_white_terrier', 117: 'whippet', 118: 'wire-haired_fox_terrier', 119: 'yorkshire_terrier'}

class App(Frame):
  def __init__(self, master):
    super().__init__(master)
    self.pack()

    self.style = Style()
    self.style.theme_use('default')

    select_image_button = Button(self, text='Select an image', command=self.select_image)
    predict_button= Button(self, text='Predict', command=self.predict)
    predict_untrained_button= Button(self, text='Predict (untrained)', command=self.predict_untrained)

    blank_img = ImageTk.PhotoImage(Image.new('RGB', (512, 512), (255, 255, 255)))
    self.image_label = Label(self, image=blank_img)
    self.image_label.image = blank_img
    self.image_label.pack(padx=10, pady=10, side=tk.LEFT)

    self.breed_labels = []
    self.prob_labels = []
    self.bar = []
    for _ in range(5):
      self.breed_labels.append(Label(self, text='Breed:'))
      self.prob_labels.append(Label(self, text='Probability:'))
      self.bar.append(Progressbar(self, orient='horizontal', mode='determinate', length=512))

    for i in range(5):
      self.breed_labels[i].pack(padx=5, pady=3, fill='both')
      self.prob_labels[i].pack(padx=5, pady=3, fill='both')
      self.bar[i].pack(padx=5, pady=10, fill='both')
    
    select_image_button.pack(padx=5, pady=5, side=tk.LEFT)
    predict_button.pack(padx=5, pady=5, side=tk.LEFT)
    predict_untrained_button.pack(padx=5, pady=5, side=tk.LEFT)

    self.filename = None

    # load saved model
    K = keras.backend.backend()
    if K == 'tensorflow':
      keras.backend.set_image_data_format('channels_last')
    self.model = keras.models.load_model(SAVED_MODEL_PATH)
    self.model_untrained = keras.models.load_model(SAVED_MODEL_UNTRAINED_PATH)

  def select_image(self):
    filetypes = (
      ('JPEG', '.jpg .jpeg'),
      ('PNG', '.png'))
    self.filename = filedialog.askopenfilename(
      title='Select an image',
      initialdir='./',
      filetypes=filetypes)

    if self.filename:
      img = Image.open(self.filename).resize((512, 512))
      img = ImageTk.PhotoImage(img)
      self.image_label.configure(image=img)
      self.image_label.image = img
      self.reset()

  def predict_common(self, model):
    self.reset()
    if self.filename:
      img = image.load_img(self.filename, target_size=(256, 256), color_mode='rgb', interpolation='nearest')
      img = image.img_to_array(img)
      img /= 255.0
      print(img)
      img = np.expand_dims(img, axis=0)
      result = model.predict(img)
      print(result)
      top5 = list(np.argsort(result, axis=1)[:,-5:][0])
      top5.reverse();
      result = list(result[0])
      top5_prob = list(map(lambda n : result[n] * 100, top5))
      top5 = list(map(lambda n : LABEL_NAMES[n].replace('_', ' ').title(), top5))
      for i, (name, prob) in enumerate(zip(top5, top5_prob)):
        self.breed_labels[i].configure(text='Breed: ' + name)
        self.prob_labels[i].configure(text='Probability: ' + str(prob) + '%')
        self.bar[i]['value'] = prob
    else:
      messagebox.showinfo('No image selected', 'Please select an image from your computer first!')

  def predict(self):
    self.predict_common(self.model)
  
  def predict_untrained(self):
    self.predict_common(self.model_untrained)

  def reset(self):
    for i in range(5):
      self.breed_labels[i].configure(text='Breed:')
      self.prob_labels[i].configure(text='Probability:')
      self.bar[i]['value'] = 0

root = tk.Tk()
root.title('Dog breed analyzer')
root.resizable(False, False)

swidth = root.winfo_screenwidth()
sheight = root.winfo_screenheight()

x_offset = int((swidth / 2) - (WIDTH / 2))
y_offset = int((sheight / 2) - (HEIGHT / 2))

root.geometry('{}x{}+{}+{}'.format(WIDTH, HEIGHT, x_offset, y_offset))
app = App(root)
app.mainloop()

