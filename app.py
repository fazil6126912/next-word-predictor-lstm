import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_num_words):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_num_words:
    token_list = token_list[-(max_num_words-1):]
  token_list = pad_sequences([token_list],maxlen=max_num_words-1,padding='pre')
  predicted = model.predict(token_list,verbose=0)
  predicted_index = np.argmax(predicted,axis = 1)[0]
  predicted_word = ''
  for word,index in tokenizer.word_index.items():
    if index == predicted_index:
      predicted_word = word
      break
  return predicted_word

st.title("Next Word Predictior")
input_text = st.text_input("Enter the sentence Words", "Today is a beautiful day and I love")
if st.button("Predict Next Word"):
  max_seq = model.input_shape[1]+1
  predicted_word = predict_next_word(model,tokenizer,input_text,max_seq)
  st.write('Next Word: ',predicted_word)