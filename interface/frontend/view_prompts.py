import json
import itertools
import textwrap
import time
from multiprocessing import Manager, Pool
import threading

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

from data_collection import show_jinja, show_text, test_template

template_column = lambda template: 'prompt:%s' % template.name

def label(db, template, idx, dataset):
    print('Labelling %s on example %d' % (template.name, idx))
    # run api here
    url, applied_template, choices, probs = test_template(template, dataset[idx])

    d = {}
    d[template_column(template)] = (choices, probs)
    db.update(d, Query().idx == idx)

def labelling_thread(db, templates, dataset):
    try:
        print('Started thread...')
        for template in templates:
            ex = Query()
            hits = db.search(~ex[template_column(template)].exists())

            print('%s, %d' % (template.name, len(hits)))
            for hit in hits:
                label(db, template, hit['idx'], dataset)
    finally:
        print('Ending thread.')

def main(state, dataset, dataset_templates, db, LABEL_FIELD = 'label'):
    # View one template
    st.header('Prompt viewer')
    st.write(dataset_templates.folder_path)

    name = st.selectbox(
        label="Choose a prompt to view",
        options=dataset_templates.all_template_names,
        index=0,
        key="prompt_select",
    )

    template = dataset_templates[name] 
    st.write('Jinja:')
    show_jinja(template.jinja)
    st.write('Answer choices:')
    if template.answer_choices:
        st.write(template.answer_choices.split(' ||| '))
    else:
        st.write(None)

    st.markdown('---')
    st.header('Start predicting!')
    run_prompt = st.button("Start predicting with selected prompt")
    run_all_prompts = st.button("Start predicting with all prompts")
    stop_prompt = st.button("Stop background predictions")

    if run_prompt or run_all_prompts:
        if run_prompt:
            this_template = dataset_templates[name]
            all_templates = [ this_template ]
        elif run_all_prompts:
            all_templates = dataset_templates.templates.values()


        if state['labelling_thread'] is None:
            # start a process
            thread = threading.Thread(target=labelling_thread, args=(db, all_templates, dataset))
            state['labelling_thread'] = thread
            thread.start()
        else:
            st.error('Already running!')


    st.markdown('---')
    st.header('Prompt labelling')
    if state['labelling_thread']:
        if state['labelling_thread'].is_alive():
            status = 'running'
        else:
            # dead thread
            state['labelling_thread'] = None
            status = 'idle'
    else:
        status = 'idle'


    st.write('Labelling status: %s' % status)


    df = []
    for n in dataset_templates.all_template_names:
        col = template_column(dataset_templates[n])
        ex = Query()
        hits = db.search(ex[col] != None)
        labelled_num = len(hits)
        df.append((n, labelled_num))

    df = pd.DataFrame(df, columns=['Prompt', 'Labelled examples'])
    st.write(df)
    refresh = st.button("Update")
