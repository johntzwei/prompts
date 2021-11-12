import streamlit as st
import pandas as pd
import numpy as np
import datasets

name = 'ag_news'
sec = 'train'
ds = datasets.load_dataset(name)
ds_sec = ds[sec]

st.title('Let\'s annotate %s:%s!' % (name, sec))
