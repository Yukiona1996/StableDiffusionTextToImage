import tkinter as tk
# import tkinter
import customtkinter as ctk 
# import customtkinter

from time import time
import os
import psutil

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

torch.cuda.empty_cache()
# print(torch.cuda.is_available())
# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
# device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
# pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token) 
pipe.to(device)

def generate(): 
    start_time = time()
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
        memory_before = get_memory_usage()
        print("Memory usage before image generation:", memory_before, "GB")
        # image = pipe(prompt.get(), guidance_scale=8.5).images[0]
        end_time = time()
        print("Image generation time:", end_time - start_time, "seconds")
        memory_after = get_memory_usage()
        print("Memory usage after image generation:", memory_after, "GB")
        
        image.save('generatedimage.png')
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)
        torch.cuda.empty_cache()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to gigabytes
    return memory_usage

trigger = ctk.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate") 
trigger.place(x=206, y=60)

app.mainloop()