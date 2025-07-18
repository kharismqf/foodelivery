import streamlit as st
from streamlit_option_menu import option_menu

# Import fungsi halaman dari folder src
# from src.predict import show_predict_page
from src.overview import show_overview
from src.eda import show_eda
from src.aboutme import show_creator
from src.predict import show_predict_page

# Sidebar menu navigation
with st.sidebar:
    page = option_menu(
        menu_title='Main Menu',
        options=['Understand the Data',
                 'Explore The Data',         
                 'Try the Model',
                 'Meet the Creator'],
        icons=['bar-chart', 'graph-up', 'cpu', 'person-circle'],
        default_index=0
    )

# Page 1: Understand the Data
if page == 'Understand the Data':
    show_overview()

# Page 2: Explore The Data
elif page == "Explore The Data":
    show_eda()

# Page 3: Try the Model
elif page == "Try the Model":
    show_predict_page()

# Page 4: Meet the Creator
elif page == 'Meet the Creator':
    from src.aboutme import show_creator
    show_creator()
