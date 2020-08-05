#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:13:39 2020

@author: mateus
"""
import tkinter as tk
from tkinter import messagebox

root= tk.Tk()
root.geometry('300x200')

def ExitApp():
    MsgBox = messagebox.askquestion('Exit App','Really Quit?',icon = 'error')
    if MsgBox == 'yes':
       root.destroy()
    else:
        tk.messagebox.showinfo('Welcome Back','Welcome back to the App')
        
buttonEg = tk.Button(root, text='Exit App',command=ExitApp)
buttonEg.pack()
  
root.mainloop()