import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from scipy.stats import t
import math
import os
import numpy as np
from dotenv import load_dotenv
import plotly.graph_objs as go
#importamos las librerias para la red bayesiana
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
#importamos las librerias de conexion a bases de datos
from sqlalchemy import create_engine
#importamos la libreria para leer el modelo de la red definitiva en formato .bif
from pgmpy.readwrite import BIFReader
#importamos los datos ya limpios de heart_desease
#heart_desease=pd.read_csv(str(os.getcwd())+"/"+"datos_para_red")
#construimos la red, con los nodos:
#modelo=BayesianNetwork([('E','A'),('E','H'),('C','EC'),('S','EC'),('T','EC'),('A','EC'),
#                        ('H','EC'),('EC','RE'),('EC','HR'),('EC','P'),('EC','AI'),('EC','SER'),
#                       ('EC','F'),('EC','D'),('S','D')])

env_path='env\\app.env'
# load env 
load_dotenv(dotenv_path=env_path)
# extract env variables
USER=os.environ.get('USUARIO')
PASSWORD=os.getenv('CLAVE')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')
print(USER)
print("el anterior es el usuario")
#Nos conectamos a la base de datos
engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(USER, PASSWORD, HOST, PORT, DBNAME))
dbConnection=engine.connect();
print("conexion a db ok!")
heart_disease=pd.read_sql("SELECT * FROM heart_disease", dbConnection)
heart_disease=heart_disease.drop(labels="index",axis=1)

#heart_disease=pd.read_csv(str(os.getcwd())+"/"+"datos_para_grafica")
#Se importan los Datos
datosMaximaFrecuenciaCardiaca=heart_disease.loc[:,["ENFERMEDAD_CARD","MAX_HEART_R"]]
datosMaximaFrecuenciaCardiacaSano=datosMaximaFrecuenciaCardiaca[["ENFERMEDAD_CARD","MAX_HEART_R"]][datosMaximaFrecuenciaCardiaca["ENFERMEDAD_CARD"]=="Sano"]
datosMaximaFrecuenciaCardiacaEnfermo=datosMaximaFrecuenciaCardiaca[["ENFERMEDAD_CARD","MAX_HEART_R"]][datosMaximaFrecuenciaCardiaca["ENFERMEDAD_CARD"]=="Enfermo"]
datosEdad=heart_disease.loc[:,["ENFERMEDAD_CARD","EDAD"]]
datosEdadSano=datosEdad[["ENFERMEDAD_CARD","EDAD"]][datosEdad["ENFERMEDAD_CARD"]=="Sano"]
datosEdadEnfermo=datosEdad[["ENFERMEDAD_CARD","EDAD"]][datosEdad["ENFERMEDAD_CARD"]=="Enfermo"]
datosPresionSanguinea=heart_disease.loc[:,["ENFERMEDAD_CARD","PRESION_SAN"]]
datosPresionSanguineaSano=datosPresionSanguinea[["ENFERMEDAD_CARD","PRESION_SAN"]][datosPresionSanguinea["ENFERMEDAD_CARD"]=="Sano"]
datosPresionSanguineaEnfermo=datosPresionSanguinea[["ENFERMEDAD_CARD","PRESION_SAN"]][datosPresionSanguinea["ENFERMEDAD_CARD"]=="Enfermo"]
datosDolorPecho=heart_disease.loc[:,["ENFERMEDAD_CARD","DOLOR_PECHO"]]
datosDolorPechoSano=datosDolorPecho[["ENFERMEDAD_CARD","DOLOR_PECHO"]][datosDolorPecho["ENFERMEDAD_CARD"]=="Sano"]
datosDolorPechoEnfermo=datosDolorPecho[["ENFERMEDAD_CARD","DOLOR_PECHO"]][datosDolorPecho["ENFERMEDAD_CARD"]=="Enfermo"]
#En el anterior bloque de codigo lo que se hace para Presion, MPC, Colesterol, y Tipo de Dolor de pecho es:
#1.Extraer los datos del DF de la variable junto a una variable que dice si la persona esta enferma o no [ENFERMEDAD_CARD], creando un nuevo DF
#2.Se filtran los datos y se crea 2 nuevos DF uno para los que estan enfermos y otro para los saludables

#Estadisticas
mediaMuestralFreqSano=datosMaximaFrecuenciaCardiacaSano["MAX_HEART_R"].mean()
mediaMuestralFreqEnfermo=datosMaximaFrecuenciaCardiacaEnfermo["MAX_HEART_R"].mean()
mediaMuestralEdadSano=datosEdadSano["EDAD"].mean()
mediaMuestralEdadEnfermo=datosEdadEnfermo["EDAD"].mean()
dfFreqSano=datosMaximaFrecuenciaCardiacaSano["MAX_HEART_R"].count()-1
dfFreqEnfermo=datosMaximaFrecuenciaCardiacaEnfermo["MAX_HEART_R"].count()-1
dfEdadSano=datosEdadSano["EDAD"].count()-1
dfEdadEnfermo=datosEdadEnfermo["EDAD"].count()-1
tFreqSano=t.ppf(1-0.05/2,dfFreqSano,loc=0, scale=1)
tFreqEnfermo=t.ppf(1-0.05/2,dfFreqEnfermo,loc=0, scale=1)
tEdadSano=t.ppf(1-0.05/2,dfEdadSano,loc=0, scale=1)
tEdadEnfermo=t.ppf(1-0.05/2,dfEdadEnfermo,loc=0, scale=1)
varianzaMuestralFreqSano=datosMaximaFrecuenciaCardiacaSano["MAX_HEART_R"].var()
varianzaMuestralFreqEnfermo=datosMaximaFrecuenciaCardiacaEnfermo["MAX_HEART_R"].var()
varianzaMuestralEdadSano=datosEdadSano["EDAD"].var()
varianzaMuestralEdadEnfermo=datosEdadEnfermo["EDAD"].var()
mediaMuestralPresionSano=datosPresionSanguineaSano["PRESION_SAN"].mean()
mediaMuestralPresionEnfermo=datosPresionSanguineaEnfermo["PRESION_SAN"].mean()
dfPresionSano=datosPresionSanguineaSano["PRESION_SAN"].count()-1
dfPresionEnfermo=datosPresionSanguineaEnfermo["PRESION_SAN"].count()-1
tPresionSano=t.ppf(1-0.05/2,dfPresionSano,loc=0, scale=1)
tPresionEnfermo=t.ppf(1-0.05/2,dfPresionEnfermo,loc=0, scale=1)
varianzaMuestralPresionSano=datosPresionSanguineaSano["PRESION_SAN"].var()
varianzaMuestralPresionEnfermo=datosPresionSanguineaEnfermo["PRESION_SAN"].var()
mediaMuestralDolorPechoSano=(datosDolorPechoSano["DOLOR_PECHO"].astype(int)).mean()
mediaMuestralDolorPechoEnfermo=(datosDolorPechoEnfermo["DOLOR_PECHO"].astype(int)).mean()
dfDolorPechoSano=datosDolorPechoSano["DOLOR_PECHO"].count()-1
dfDolorPechoEnfermo=datosDolorPechoEnfermo["DOLOR_PECHO"].count()-1
tDolorPechoSano=t.ppf(1-0.05/2,dfDolorPechoSano,loc=0, scale=1)
tDolorPechoEnfermo=t.ppf(1-0.05/2,dfDolorPechoEnfermo,loc=0, scale=1)
varianzaMuestralDolorPechoSano=(datosDolorPechoSano["DOLOR_PECHO"].astype(int)).var()
varianzaMuestralDolorPechoEnfermo=(datosDolorPechoEnfermo["DOLOR_PECHO"].astype(int)).var()
#En el anterior bloque de codigo lo que se hace para Presion, MPC, Colesterol, y Tipo de Dolor de pecho es:
#1. Se calcula la media muestral para el grupo de datos de personas saludable y enfermas
#2. Se calcula la varianza muestral para el grupo de datos de personas saludable y enfermas
#3. Dado que se piensa hacer pruebas t simples se determina que los grados de libertad es n_ij-1 donde i pertence Presion, MPC, Colesterol, Tipo de Dolor de pecho, 
#y j pertence a enfermo, saludable
#4. Se calcula la t para cada muestra con un alpha de 0.0001
upperIPFreqSano=mediaMuestralFreqSano+tFreqSano*math.sqrt(varianzaMuestralFreqSano*(1+(1/(dfFreqSano+1))))
lowerIPFreqSano=mediaMuestralFreqSano-tFreqSano*math.sqrt(varianzaMuestralFreqSano*(1+(1/(dfFreqSano+1))))
upperIPFreqEnfermo=mediaMuestralFreqEnfermo+tFreqEnfermo*math.sqrt(varianzaMuestralFreqEnfermo*(1+(1/(dfFreqEnfermo+1))))
lowerIPFreqEnfermo=mediaMuestralFreqEnfermo-tFreqEnfermo*math.sqrt(varianzaMuestralFreqEnfermo*(1+(1/(dfFreqEnfermo+1))))
upperIPEdadSano=mediaMuestralEdadSano+tEdadSano*math.sqrt(varianzaMuestralEdadSano*(1+(1/(dfEdadSano+1))))
lowerIPEdadSano=mediaMuestralEdadSano-tEdadSano*math.sqrt(varianzaMuestralEdadSano*(1+(1/(dfEdadSano+1))))
upperIPEdadEnfermo=mediaMuestralEdadEnfermo+tEdadEnfermo*math.sqrt(varianzaMuestralEdadEnfermo*(1+(1/(dfEdadEnfermo+1))))
lowerIPEdadEnfermo=mediaMuestralEdadEnfermo-tEdadEnfermo*math.sqrt(varianzaMuestralEdadEnfermo*(1+(1/(dfEdadEnfermo+1))))
upperIPPresionSano=mediaMuestralPresionSano+tPresionSano*math.sqrt(varianzaMuestralPresionSano*(1+(1/(dfPresionSano+1))))
lowerIPPresionSano=mediaMuestralPresionSano-tPresionSano*math.sqrt(varianzaMuestralPresionSano*(1+(1/(dfPresionSano+1))))
upperIPPresionEnfermo=mediaMuestralPresionEnfermo+tPresionEnfermo*math.sqrt(varianzaMuestralPresionEnfermo*(1+(1/(dfPresionEnfermo+1))))
lowerIPPresionEnfermo=mediaMuestralPresionEnfermo-tPresionEnfermo*math.sqrt(varianzaMuestralPresionEnfermo*(1+(1/(dfPresionEnfermo+1))))
upperIPDolorPechoSano=math.ceil(mediaMuestralDolorPechoSano+tDolorPechoSano*math.sqrt(varianzaMuestralDolorPechoSano*(1+(1/(dfPresionSano+1)))))
lowerIPDolorPechoSano=math.floor(mediaMuestralDolorPechoSano-tDolorPechoSano*math.sqrt(varianzaMuestralDolorPechoSano*(1+(1/(dfPresionSano+1)))))
upperIPDolorPechoEnfermo=math.ceil(mediaMuestralDolorPechoEnfermo+tDolorPechoEnfermo*math.sqrt(varianzaMuestralDolorPechoEnfermo*(1+(1/(dfPresionEnfermo+1)))))
lowerIPDolorPechoEnfermo=math.floor(mediaMuestralDolorPechoEnfermo-tDolorPechoEnfermo*math.sqrt(varianzaMuestralDolorPechoEnfermo*(1+(1/(dfPresionEnfermo+1)))))
#En el anterior bloque de codigo lo que se hace para Presion, MPC, Colesterol, y Tipo de Dolor de pecho es:
#1. Se generan el limite tanto superior como inferior del interalo de confianza para todas las combinaciones ij donde i pertence Presion, MPC, Colesterol, Tipo de Dolor de pecho, 
#y j pertence a enfermo, saludable

#modelo.fit(data=heart_desease,estimator=MaximumLikelihoodEstimator)
#cargamos el modelo de la red definitiva
reader = BIFReader("modelo_def.bif")
modelo = reader.get_model(state_name_type=float)#ojo con el argumento, es para que no se conviertan los numeros a strings
infer=VariableElimination(modelo)


#creamos el objeto dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Heart Disease'
server = app.server

app.layout = html.Div([
    html.Div(children=[
    html.Div(children=[html.Img(src="assets/University_of_Los_Andes_logo.png",style={'width': '40%', 'float': 'left','padding': '5px'})],
    style={'width': '10%', 'display': 'inline-block'}),
    html.Div(children=[html.H1("Predictor de enfermedades cardiacas uniandes",style={'textAlign': 'left','color': 'black', 'fontSize': 70}
    ),],style={'width': '90%', 'display': 'inline-block'}),    
    #html.H1("Gráficas",
    #),
    
    #html.Img(src="assets/University_of_Los_Andes_logo.png",style={'width': '10%', 'float': 'right','padding': '10px'}),
    html.Br(),html.Label("Ingrese a cotinuacion algunos datos de triage del paciente para obtener una visualizacion de su sitación",style={'textAlign': 'left','color': 'black', 'fontSize': 30}),
    html.Br(),
    html.Div(children=[html.Label("Máxima frecuencia cardíaca",style={'textAlign': 'left','color': 'black', 'fontSize': 20}),
    dcc.Input(placeholder="Valor sin unidades",
              type="number",
              value=0,
              id="MFCGraficas",style={'width': '100%',"display":"flex", "justifyContent":'center'}
    )],style={'width': '50%', 'display': 'inline-block'}),
    html.Div(children=[html.Label("Edad",style={'textAlign': 'left','color': 'black', 'fontSize': 20}),
    dcc.Input(placeholder="Valor sin unidades",
              type="number",
              value=0,
              id="EdadGraficas",style={'width': '100%',"display":"flex", "justifyContent":'center'}
    )],style={'width': '50%', 'display': 'inline-block'}),
  
    
    
    
    html.Br(),html.Br(),

    html.Div(children=[html.Label("Presión sanguinea", style={'textAlign': 'left','color': 'black', 'fontSize': 20}),
    dcc.Input(placeholder="Valor sin unidades",
              type="number",
              value=0,
              id="PresionGraficas",style={'width': '100%',"display":"flex", "justifyContent":'center'}
    )],style={'width': '50%', 'display': 'inline-block'}),

    html.Div(children=[html.Label(['Tipo de dolor en el pecho: 1=Angina tipica, 2=Angina no tipica, 3=dolor no anginal, 4=asintomatico'],
    style={'textAlign': 'left','color': 'black', 'fontSize': 20}),
    dcc.Input(placeholder="Valor sin unidades",
              type="number",
              value=0,
              id="TipoDolorGraficas",style={'width': '100%',"display":"flex", "justifyContent":'center'}
    )],style={'width': '50%', 'display': 'inline-block'}),






    
    
    
    html.Div([
    html.Div(
        dcc.Graph(id="graficas")
    ),
    dcc.Slider(
        id='graph-selector',
        min=1,
        max=3,
        value=1,
        marks={
            1: 'Graph 1',
            2: 'Graph 2',
            3: 'Graph 3'
        },
        step=None,
        included=False
    )
]),
    ]
    ),
    html.Br(),
    html.Div(children=[
        html.H1("Ingrese los datos detallados de la historia clinica del paciente para hacer una prediccion", style={'font-size':25}),
        html.Label('Edad'),
        dcc.Dropdown(["Mayor de 60 años","Menor de 60 años"],id="edad"),
        html.Br(),
        html.Label('Sexo'),
        dcc.Dropdown(["Hombre","Mujer"],id="sexo"),
        html.Br(),
        html.Label('¿El nivel de colesterol total es superior a 200 mg/dl?'),
        dcc.Dropdown(["Si","No"],id="colesterol"),
        html.Br(),
        html.Label('¿El paciente ha sido diagnosticado diabetico?'),
        dcc.Dropdown(["Si","No"],id="diabetes"),
        html.Br(),
        html.Label('¿El paciente ha sido hipertenso (presion sanguinea mayor a 130 mm de Hg)?'),
        dcc.Dropdown(["Si","No"],id="presion"),
        html.Br(),
        html.Label('¿El paciente presenta dolor en el pecho?'),
        dcc.Dropdown(["Si","No"],id="dolor_pecho"),
        html.Br(),
        html.Label('¿En el examen de esfuerzo, el paciente presenta angina inducida?'),
        dcc.Dropdown(["Si","No"],id="angina_inducida")
        



        
    ],style={'width': '50%', 'display': 'inline-block'}),
    html.Div(children=[
        html.Label('¿Talasemia?'),
        dcc.Dropdown(["Sufre Talasemia","No sufre talasemia"],id="talasemia"),
        html.Br(),
        html.Label('Ingrese aqui el resultado de la depresion en el pico de la curva del electrocardiograma'),
        dcc.Dropdown(["Depresion de pico de ejercicio normal","Depresion de pico de ejercicio >1mm"],id="ST_depresion"),
        html.Br(),
        html.Label('La forma de la onda del electrocardiograma es normal'),
        dcc.Dropdown(["Si","No"],id="electro_card"),
        html.Br(),
        html.Label('¿La maxima frecuencia cardiaca es menor a 150?'),
        dcc.Dropdown(["Si","No"],id="max_h_rate"),
        html.Br(),
        html.Label('¿La pendiente de pico de ejercicio es creciente?'),
        dcc.Dropdown(["Si","No"],id="pendiente"),
        html.Br(),
        html.Label('¿El examen de fluoroscopia muestra algun vaso sanguineo resaltado?'),
        dcc.Dropdown(["Si","No"],id="fluor"),
        html.Br(),
        #html.Label('Calcular probabilidad de enfermedad cardiaca'),
        html.Div(id="probabilidad",style={'textAlign': 'center','color': 'black', 'fontSize': 15})
        

        
        




    ],style={'width': '50%', 'display': 'inline-block'}),
    html.Div(children=[html.Br(), html.Button('Calcular probabilidad de enfermedad cardiaca', id='calcular', n_clicks=0,style={'font-size': '12px', 
    'width': '100%', 'display': 'inline-block', 
    'margin-bottom': '10px', 'margin-right': '5px', 'height':'37px', 'verticalAlign': 'top'})])
   
])
#Queremos que la funcion cambien las graficas por eso los outputs son las figures de las 3 graficas
#De ipunts tenemos el boton
#Y states los valores que inserta el usuario


@app.callback(
    Output('graficas', 'figure'),
    [Input('graph-selector', 'value')],
    [State("MFCGraficas", "value"),State("EdadGraficas", "value"),State("PresionGraficas", "value"),State("TipoDolorGraficas", "value")]
)
def update_graph(value, MFC, Edad, Presion, TipoDolor):
    if value == 1:
        if MFC==0 and Edad==0 and Presion==0 and TipoDolor==0:
            scatterSano1=go.Scatter(x=datosEdadSano["EDAD"], y=datosMaximaFrecuenciaCardiacaSano["MAX_HEART_R"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo1=go.Scatter(x=datosEdadEnfermo["EDAD"], y=datosMaximaFrecuenciaCardiacaEnfermo["MAX_HEART_R"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph1=go.Layout(title="Scatter de la Máxima Frecuencia Cardiaca y Edad", xaxis=dict(title="Edad"), yaxis=dict(title="Maxima Frecuencia Cardiaca [bpm]"))
            rectSano1=go.Scatter(x=[lowerIPEdadSano, lowerIPEdadSano, upperIPEdadSano, upperIPEdadSano, lowerIPEdadSano], y=[lowerIPFreqSano, upperIPFreqSano, upperIPFreqSano, lowerIPFreqSano, lowerIPFreqSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo1=go.Scatter(x=[lowerIPEdadEnfermo, lowerIPEdadEnfermo, upperIPEdadEnfermo, upperIPEdadEnfermo, lowerIPEdadEnfermo], y=[lowerIPFreqEnfermo, upperIPFreqEnfermo, upperIPFreqEnfermo, lowerIPFreqEnfermo, lowerIPFreqEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            graph1=go.Figure(data=[scatterSano1, scatterEnfermo1, rectSano1, rectEnfermo1], layout=layoutGraph1)
            return graph1
        else:
            scatterSano1=go.Scatter(x=datosEdadSano["EDAD"], y=datosMaximaFrecuenciaCardiacaSano["MAX_HEART_R"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo1=go.Scatter(x=datosEdadEnfermo["EDAD"], y=datosMaximaFrecuenciaCardiacaEnfermo["MAX_HEART_R"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph1=go.Layout(title="Scatter de la Máxima Frecuencia Cardiaca y Edad", xaxis=dict(title="Edad"), yaxis=dict(title="Maxima Frecuencia Cardiaca [bpm]"))
            rectSano1=go.Scatter(x=[lowerIPEdadSano, lowerIPEdadSano, upperIPEdadSano, upperIPEdadSano, lowerIPEdadSano], y=[lowerIPFreqSano, upperIPFreqSano, upperIPFreqSano, lowerIPFreqSano, lowerIPFreqSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo1=go.Scatter(x=[lowerIPEdadEnfermo, lowerIPEdadEnfermo, upperIPEdadEnfermo, upperIPEdadEnfermo, lowerIPEdadEnfermo], y=[lowerIPFreqEnfermo, upperIPFreqEnfermo, upperIPFreqEnfermo, lowerIPFreqEnfermo, lowerIPFreqEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            puntoPaciente1=go.Scatter(x=[Edad],y=[MFC], mode="markers", marker=dict(symbol="x",color="black",size=10), name="Paciente Actual")
            graph1=go.Figure(data=[scatterSano1, scatterEnfermo1, puntoPaciente1, rectSano1, rectEnfermo1], layout=layoutGraph1)
            return graph1
    elif value ==2:
        if MFC==0 and Edad==0 and Presion==0 and TipoDolor==0:
            scatterSano2=go.Scatter(x=datosEdadSano["EDAD"], y=datosPresionSanguineaSano["PRESION_SAN"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo2=go.Scatter(x=datosEdadEnfermo["EDAD"], y=datosPresionSanguineaSano["PRESION_SAN"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph2=go.Layout(title="Scatter de la Presión Sanguinea y Edad", xaxis=dict(title="Edad"), yaxis=dict(title="Presión Sanguinea [mm Hg]"))
            rectSano2=go.Scatter(x=[lowerIPEdadSano, lowerIPEdadSano, upperIPEdadSano, upperIPEdadSano, lowerIPEdadSano], y=[lowerIPPresionSano, upperIPPresionSano, upperIPPresionSano, lowerIPPresionSano, lowerIPPresionSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo2=go.Scatter(x=[lowerIPEdadEnfermo, lowerIPEdadEnfermo, upperIPEdadEnfermo, upperIPEdadEnfermo, lowerIPEdadEnfermo], y=[lowerIPPresionEnfermo, upperIPPresionEnfermo, upperIPPresionEnfermo, lowerIPPresionEnfermo, lowerIPPresionEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            graph2=go.Figure(data=[scatterSano2, scatterEnfermo2, rectSano2, rectEnfermo2], layout=layoutGraph2)
            return graph2
        else:
            scatterSano2=go.Scatter(x=datosEdadSano["EDAD"], y=datosPresionSanguineaSano["PRESION_SAN"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo2=go.Scatter(x=datosEdadEnfermo["EDAD"], y=datosPresionSanguineaSano["PRESION_SAN"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph2=go.Layout(title="Scatter de la Presión Sanguinea y Edad", xaxis=dict(title="Edad"), yaxis=dict(title="Presión Sanguinea [mm Hg]"))
            rectSano2=go.Scatter(x=[lowerIPEdadSano, lowerIPEdadSano, upperIPEdadSano, upperIPEdadSano, lowerIPEdadSano], y=[lowerIPPresionSano, upperIPPresionSano, upperIPPresionSano, lowerIPPresionSano, lowerIPPresionSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo2=go.Scatter(x=[lowerIPEdadEnfermo, lowerIPEdadEnfermo, upperIPEdadEnfermo, upperIPEdadEnfermo, lowerIPEdadEnfermo], y=[lowerIPPresionEnfermo, upperIPPresionEnfermo, upperIPPresionEnfermo, lowerIPPresionEnfermo, lowerIPPresionEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            puntoPaciente2=go.Scatter(x=[Edad],y=[Presion], mode="markers", marker=dict(symbol="x",color="black",size=10), name="Paciente Actual")
            graph2=go.Figure(data=[scatterSano2, scatterEnfermo2,puntoPaciente2 ,rectSano2, rectEnfermo2], layout=layoutGraph2)
            return graph2
    elif value ==3:
        if MFC==0 and Edad==0 and Presion==0 and TipoDolor==0:
            scatterSano3=go.Scatter(x=datosPresionSanguineaSano["PRESION_SAN"], y=datosDolorPechoSano["DOLOR_PECHO"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo3=go.Scatter(x=datosPresionSanguineaEnfermo["PRESION_SAN"], y=datosDolorPechoEnfermo["DOLOR_PECHO"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph3=go.Layout(title="Scatter de la Presión Sanguinea y el tipo de Dolor de Pecho", xaxis=dict(title="Presion sanguinea [mm HG]"), yaxis=dict(title="Tipo de dolor de pecho"))
            rectSano3=go.Scatter(x=[lowerIPPresionSano, lowerIPPresionSano, upperIPPresionSano, upperIPPresionSano, lowerIPPresionSano], y=[lowerIPDolorPechoSano, upperIPDolorPechoSano, upperIPDolorPechoSano, lowerIPDolorPechoSano, lowerIPDolorPechoSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo3=go.Scatter(x=[lowerIPPresionEnfermo, lowerIPPresionEnfermo, upperIPPresionEnfermo, upperIPPresionEnfermo, lowerIPPresionEnfermo], y=[lowerIPDolorPechoEnfermo, upperIPDolorPechoEnfermo, upperIPDolorPechoEnfermo, lowerIPDolorPechoEnfermo, lowerIPDolorPechoEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            graph3=go.Figure(data=[scatterSano3, scatterEnfermo3, rectSano3, rectEnfermo3], layout=layoutGraph3)
            return graph3
        else:
            scatterSano3=go.Scatter(x=datosPresionSanguineaSano["PRESION_SAN"], y=datosDolorPechoSano["DOLOR_PECHO"], mode="markers", marker=dict(color='rgb(51, 255, 255)'), name="Saludables")
            scatterEnfermo3=go.Scatter(x=datosPresionSanguineaEnfermo["PRESION_SAN"], y=datosDolorPechoEnfermo["DOLOR_PECHO"], mode="markers", marker=dict(color='rgb(255, 51, 51)'), name= "Enfermos")
            layoutGraph3=go.Layout(title="Scatter de la Presión Sanguinea y el tipo de Dolor de Pecho", xaxis=dict(title="Presion sanguinea [mm HG]"), yaxis=dict(title="Tipo de dolor de pecho"))
            rectSano3=go.Scatter(x=[lowerIPPresionSano, lowerIPPresionSano, upperIPPresionSano, upperIPPresionSano, lowerIPPresionSano], y=[lowerIPDolorPechoSano, upperIPDolorPechoSano, upperIPDolorPechoSano, lowerIPDolorPechoSano, lowerIPDolorPechoSano], fill="toself", name= "Predicción Saludables", marker=dict(color='rgb(204, 255, 255)'))
            rectEnfermo3=go.Scatter(x=[lowerIPPresionEnfermo, lowerIPPresionEnfermo, upperIPPresionEnfermo, upperIPPresionEnfermo, lowerIPPresionEnfermo], y=[lowerIPDolorPechoEnfermo, upperIPDolorPechoEnfermo, upperIPDolorPechoEnfermo, lowerIPDolorPechoEnfermo, lowerIPDolorPechoEnfermo], fill="toself", name="Predicción Enfermos", marker=dict(color='rgb(255, 204, 204)'))
            puntoPaciente3=go.Scatter(x=[Presion],y=[TipoDolor], mode="markers", marker=dict(symbol="x",color="black",size=10), name="Paciente Actual")
            graph3=go.Figure(data=[scatterSano3, scatterEnfermo3,puntoPaciente3, rectSano3, rectEnfermo3], layout=layoutGraph3)
            return graph3

#Con los intervalos previos y los datos de las personas saludables y enfermas se generan los graficos y el cuadro de IC, si el boton se presiona se capta los valores en la input box y se añade ese valor al grafico como una cruz negra

@app.callback(
    Output("probabilidad","children"),
    [Input('calcular','n_clicks'),
    Input('edad','value'),
    Input('sexo','value'),
    Input('colesterol','value'),
    Input('diabetes','value'),
    Input('presion','value'),
    Input('dolor_pecho','value'),
    Input('angina_inducida','value'),
    Input('talasemia','value'),
    Input('ST_depresion','value'),
    Input('electro_card','value'),
    Input('max_h_rate','value'),
    Input('pendiente','value'),
    Input('fluor','value'),
   ]
)
def estimar_enfermedad_card(n_clicks,edad,sexo,colesterol,diabetes,presion,dolor_pecho,angina_inducida,talasemia,
ST_depresion,electro_card,max_h_rate,pendiente,fluor):
    if n_clicks==0:#inicializa el mensaje en vacio
        return "Ingrese la informacion del paciente en las casillas y de clic en el boton de calcular"
    else:
        evidencia={}#creamos un diccionario vacio para la evidencia
        #verificamos el valor de la edad
        if edad=="Mayor de 60 años":
            evidencia["E"]=1
        elif edad=="Menor de 60 años":
            evidencia["E"]=0
        #verificamos el sexo
        if sexo=="Hombre":
            evidencia["S"]=1
        elif sexo=="Mujer":
            evidencia["S"]=0
        #verificamos colesterol:
        if colesterol=="Si":
            evidencia["C"]=1
        elif colesterol=="No":
            evidencia["C"]=0
        #verificamos la diabetes:
        if diabetes=="Si":
            evidencia["A"]=1
        elif diabetes=="No":
            evidencia["A"]=0
        #verificamos la presion arterial:
        if presion=="Si":
            evidencia["P"]=1
        elif presion=="No":
            evidencia["P"]=0
        #verificamos el dolor de pecho:
        if dolor_pecho=="Si":
            evidencia["D"]=1
        elif dolor_pecho=="No":
            evidencia["D"]=0
        #verificamos la angina inducida
        if angina_inducida=="Si":
            evidencia["AI"]=1
        elif angina_inducida=="No":
            evidencia["AI"]=0
        #verificamos la talasemia
        if talasemia=="Sufre Talasemia":
            evidencia["T"]=1
        elif talasemia=="No sufre talasemia":
            evidencia["T"]=0
        #verificamos st_depresion
        if ST_depresion=="Depresion de pico de ejercicio normal":
            evidencia["SER"]=0
        elif ST_depresion=="Depresion de pico de ejercicio >1mm":
            evidencia["SER"]=1
        #verificamos la forma de la onda del electrocardiograma:
        if electro_card=="Si":
            evidencia.setdefault("RE", 0)
            #evidencia['RE']==0
        elif electro_card=="No":
            evidencia.setdefault("RE", 1)
            #evidencia['RE']==1
        #verificamos la maxima frecuencia cardiaca
        if max_h_rate=="Si":
            evidencia["HR"]=1
        elif max_h_rate=="No":
            evidencia["HR"]=0
        #verificamos la pendiente del electro:
        if pendiente=="Si":
            evidencia["P"]=0
        elif pendiente=="No":
            evidencia["P"]=1
        #verificmos el examen de fluoresencia:
        if fluor=="Si":
            evidencia["F"]=1
        elif fluor=="No":
            evidencia["F"]=0
        #hacemos la inferencia de la probabilidad
        caso_a = infer.query(["EC"] , evidence =evidencia)
        print(evidencia)
        print(caso_a)
        respuesta="La probabilidad de que el paciente tenga una enfermedad cardiaca es de {}".format(np.round(caso_a.values[1],2))
        return respuesta
    
if __name__ == '__main__':
    app.run_server(debug=True)
