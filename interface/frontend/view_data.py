import json
import itertools
import textwrap
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from datasets import get_dataset_infos
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import DjangoLexer

from tinydb import TinyDB, Query

from promptsource.session import _get_state
from promptsource.templates import DatasetTemplates, Template, TemplateCollection
from promptsource.utils import (
    get_dataset,
    get_dataset_confs,
    list_datasets,
    removeHyphen,
    renameDatasetColumn,
    render_features,
)

def main(state, dataset, db, LABEL_FIELD = 'label'):
    step = st.selectbox('How many to display on this page?', (10, 20, 40, 80))
    start = st.slider('Database index', min_value=0, max_value=len(db)+1, step=step)

    end = start + step
    st.write('Showing rows %d to %d:' % (start, end))
    for i, row in enumerate(itertools.islice(db, start, end)):
        st.write('index: ', (start + i))

        ex = dataset[row['idx']].copy()
        del(ex[LABEL_FIELD])
        st.write(ex)

        st.write(row)
        st.markdown("""---""")
