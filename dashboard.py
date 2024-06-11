import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")
st.set_page_config(page_title="Spaceship Titanic", layout="wide", page_icon=":bar_chart:")

st.title("Spaceship Titanic")
st.subheader("Dashboard Completo")
st.dataframe(df)


# Sidebar
st.sidebar.header('Filtros')
planet = st.sidebar.multiselect(
    "Selecione o planeta",
    options=df["HomePlanet"].unique(),
    default=df["HomePlanet"].unique()
)

destiny = st.sidebar.multiselect(
    "Destino",
    options=df["Destination"].unique(),
    default=df["Destination"].unique()
)

vip = st.sidebar.multiselect(
    "VIP",
    options=[True, False],
    default=[True, False]
)

df_selection = df[
    (df["HomePlanet"].isin(planet)) &
    (df["Destination"].isin(destiny)) &
    (df["VIP"].isin(vip))
]

st.subheader("Dashboard Filtrado")
st.dataframe(df_selection)


# Mainpage

# Vendas
st.title("Vendas")
st.markdown("##")

roomservice_sales =  df_selection["RoomService"].sum()
foodcourt_sales = df_selection["FoodCourt"].sum()
shopping_sales = df_selection["ShoppingMall"].sum()
spa_sales = df_selection["Spa"].sum()
vrdeck_sales = df_selection["VRDeck"].sum()

total_sales = roomservice_sales + foodcourt_sales + shopping_sales + spa_sales + vrdeck_sales

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Categorias")
    st.write("Serviço de Quarto: $"f"{roomservice_sales:,.2f}")
    st.write("Praça de Alimentação: $"f"{foodcourt_sales:,.2f}")
    st.write("Shopping: $"f"{shopping_sales:,.2f}")
    st.write("Spa: $"f"{spa_sales:,.2f}")
    st.write("Realidade Virtual: $"f"{vrdeck_sales:,.2f}")

with right_column:
    st.subheader("Total")
    st.write("$"f"{total_sales:,.2f}")

# Gráfico de pizza
labels = ['Serviço de Quarto', 'Praça de Alimentação', 'Shopping', 'Spa', 'Realidade Virtual']
sizes = [roomservice_sales, foodcourt_sales, shopping_sales, spa_sales, vrdeck_sales]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice

fig1, ax1 = plt.subplots(figsize=(2, 2))
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'color':"w", 'fontsize':5})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Configurar fundo transparente
fig1.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0)

st.pyplot(fig1)


# Clientes
st.title("Tripulação")
st.markdown("##")

person_count = len(df_selection)
age_mean = df_selection["Age"].mean()
cryosleep_count = df_selection["CryoSleep"].value_counts()

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Informações")
    st.write("Idade Média: "f"{age_mean:,.2f}")
    st.write("Quantidade de Tripulantes: "f"{person_count}")

with right_column:
    st.subheader("Sono Criogênico")
    st.write(cryosleep_count)
