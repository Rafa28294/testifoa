import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
def convert_to_excel(df2):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df2.to_excel(writer, sheet_name="data",index=False)
    writer.close()
    return output.getvalue()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2020/06/30/14/05/botany-5356475_1280.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def fig1(df1):
    fig1 = sns.pairplot(df1, hue='class', diag_kind='kde')
    return fig1

def fig2(df1):
    fig2, axes2 = plt.subplots(2, 2)
    sns.violinplot(x='class', y='sepal length',data=df1,hue="class", ax=axes2[0,0])
    sns.violinplot(x='class', y='sepal width', data=df1,hue="class", ax=axes2[0,1])
    sns.violinplot(x='class', y='petal length', data=df1,hue="class", ax=axes2[1,0])
    sns.violinplot(x='class', y='petal width', data=df1,hue="class", ax=axes2[1,1])
    plt.tight_layout()
    return fig2

def fig3(df1):
    fig3, axes = plt.subplots()
    sns.countplot(data=df1, x='sepal width', hue='class', ax=axes)
    return fig3

def fig4(df1):
    fig4 = sns.jointplot(x='petal length',y='petal width',hue='class',data=df1)
    return fig4


def main():
    st.set_page_config(page_title='Analisi', layout='wide')
    st.title("Analisi Dati")

    add_bg_from_url() 

    uploaded_file = st.file_uploader("Scegli un file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        df1 = df.copy()
        df1.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        st.dataframe(df1)

        tab1, tab2, tab3 = st.tabs(['Analisi', 'Prediction', 'Drag&Drop'])

        with tab1:
            st.header('Analisi Esplorativa')

            if st.button('Statistiche', help="Process Dataframe"):
                st.subheader('📈 Statistiche Descrittive')
                st.dataframe(df1.describe())

            if st.button('Grafici', help="Process Dataframe"):
                st.subheader('🎨 Pairplot delle Variabili Selezionate')
                st.pyplot(fig1(df1))

                st.subheader('🎨 Violinplot delle Variabili Selezionate')
                st.pyplot(fig2(df1))

                st.subheader('🎨 Countplot delle Variabili Selezionate')
                st.pyplot(fig3(df1))

                st.subheader('🎨 Jointplot delle Variabili Selezionate')            
                st.pyplot(fig4(df1))

        with tab2:
            st.header('Selezionare le variabili')

            sepal_length = st.slider('sepal length',0.0, 10.0, 3.0, 0.1 )
            sepal_width = st.slider('sepal width',0.0, 5.0, 2.5, 0.1 )
            petal_length = st.slider('petal length',0.0, 8.0, 3.5, 0.1 )
            petal_width =  st.slider('petal width',0.0, 4.0, 2.0, 0.1 )

            data = {
                    "sepal length": [sepal_length],
                    "sepal width": [sepal_width],
                    "petal length": [petal_length],
                    "petal width": [petal_width]
                    }

            input_df = pd.DataFrame(data)

            modello_data = st.file_uploader("Scegli un modello di inferenza")
            if modello_data is not None:
                loaded_model = joblib.load(modello_data)
                if st.button('Prediction', help="Process Dataframe"):        
                        res = loaded_model.predict(input_df).astype(int)[0]
                        classes = {0:'setosa',
                                1:'versicolor',
                                2: 'virginica'
                                }
                        y_pred = classes[res]
                        st.success(y_pred)

        with tab3:
            file_verificare = st.file_uploader("Carica un file",type={"xlsx"})
            if file_verificare is not None:
                df_v = pd.read_excel(file_verificare)
                df2 = df_v.copy()
                st.dataframe(df2)

                m_data = st.file_uploader("Scegli un modello di predizione")
                if m_data is not None:
                    model_caricato = joblib.load(m_data)
                    if st.button('Predizione', help="Process Dataframe"): 
                        predic = model_caricato.predict(df2).astype(int)
                        classes = {0:'Setosa',
                                1:'Versicolor',
                                2: 'Virginica'
                                }
                        df2['pred'] = predic
                        df2['pred'] = df2['pred'].map(classes)
                        st.dataframe(df2)

                st.download_button(
                                label="download as Excel-file",
                                data=convert_to_excel(df2),
                                file_name="risultati_pred.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="excel_download",
                                )

if __name__ == "__main__":
    main()