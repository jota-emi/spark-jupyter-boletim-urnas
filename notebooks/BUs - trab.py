# 0. Imports
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum, countDistinct, round, max, count, desc
import pyspark.sql.functions as F

# 1. Iniciando Sessão Spark
spark = SparkSession.builder \
        .appName("trab-eng-dados") \
        .master("spark://spark:7077") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

# 2. Exploração Inicial dos Dados
## 2.1 Carregando os dados
data_raw = spark.read.csv("hdfs://namenode:9000/data", header=True, inferSchema=True, encoding="latin1", sep=";")

# 2.2 Conhecendo o Dataset
## 2.2.1 Shape
row_count = data_raw.count()
column_count = len(data_raw.columns)
print(f"Qtd Linhas: {row_count} \nQtd Colunas: {column_count}")

# 2.2.2 Tipo dos dados
data_raw.printSchema()

# 2.2.3 Registros
data_raw.show(3, truncate=False, vertical=True)

# 2.2.4 Describe
data_raw.describe().show(truncate=False, vertical=True)

# 2.2.5 Quantidade de valores nulos
# Calculate the count of null values in each column
null_counts = [sum(col(column_name).isNull().cast("integer")).alias(column_name) for column_name in data_raw.columns]

# Create a DataFrame with the null value counts
null_counts_df = data_raw.select(null_counts)

# Display the null value counts
null_counts_df.show(vertical=True)

# 2.2.6 Quantidade de valores distintos
distinct_counts_df = data_raw.agg(*(countDistinct(col(c)).alias(c) for c in data_raw.columns))
distinct_counts_df.show(truncate=False, vertical=True)

# 3. Limpeza e Transformação dos dados
## 3.1 Deletando amostras com valores nulos
# Drop rows with null values
data_cleaned = data_raw.na.drop()

# Calculate the count of null values in each column
null_counts = [sum(col(column_name).isNull().cast("integer")).alias(column_name) for column_name in data_cleaned.columns]

# Create a DataFrame with the null value counts
null_counts_df = data_cleaned.select(null_counts)

# Display the null value counts
null_counts_df.show(vertical=True)

# 4. Análise Exploratória
## 4.1 Taxa de comparecimento às eleições
## 4.1.1 - Por Seção Eleitoral
taxa_comp_secao = (data_cleaned
                 .groupBy("NM_MUNICIPIO", "NR_ZONA", "NR_SECAO")
                 .agg(
                     max("QT_APTOS").alias("QT_APTOS"),
                     max("QT_COMPARECIMENTO").alias("QT_COMPARECIMENTO"),
                     max("QT_ABSTENCOES").alias("QT_ABSTENCOES")
                    )
                 .withColumn("TAXA_COMPARECIMENTO", round((col("QT_COMPARECIMENTO") / col("QT_APTOS") * 100), 2))
                 .orderBy("NR_ZONA", "NR_SECAO")
                )
taxa_comp_secao.show(truncate=False)

# 4.1.2 - Por Zona Eleitoral
taxa_comp_zona = (taxa_comp_secao
                     .groupBy("NR_ZONA")
                     .agg(
                         sum("QT_APTOS").alias("QT_APTOS"),
                         sum("QT_COMPARECIMENTO").alias("QT_COMPARECIMENTO"),
                         sum("QT_ABSTENCOES").alias("QT_ABSTENCOES")
                        )
                     .withColumn("TAXA_COMPARECIMENTO", round((col("QT_COMPARECIMENTO") / col("QT_APTOS") * 100), 2))
                     .withColumn("TAXA_ABSTENCOES", round((col("QT_ABSTENCOES") / col("QT_APTOS") * 100), 2))
                     .orderBy("NR_ZONA")
                    )
taxa_comp_zona.show(truncate=False)

# 4.1.3 - Por Município
taxa_comp_mun = (taxa_comp_secao
                 .groupBy("NM_MUNICIPIO")
                 .agg(
                     sum("QT_APTOS").alias("QT_APTOS"),
                     sum("QT_COMPARECIMENTO").alias("QT_COMPARECIMENTO"),
                     sum("QT_ABSTENCOES").alias("QT_ABSTENCOES")
                    )
                 .withColumn("TAXA_COMPARECIMENTO", round((col("QT_COMPARECIMENTO") / col("QT_APTOS") * 100), 2))
                 .withColumn("TAXA_ABSTENCOES", round((col("QT_ABSTENCOES") / col("QT_APTOS") * 100), 2))
                 .orderBy("NM_MUNICIPIO")
                )
taxa_comp_mun.show(truncate=False)

# Temporary ordered df
temp_df = taxa_comp_mun.orderBy(col("TAXA_ABSTENCOES").desc()).toPandas()
# Fig
fig, ax = plt.subplots(figsize=(5, 25))
# Plot
temp_df['TAXA_COMPARECIMENTO'] = 100
sns.barplot(data=temp_df, x="TAXA_COMPARECIMENTO", y="NM_MUNICIPIO", color="skyblue", label="Comparecimento", ax=ax)
sns.barplot(data=temp_df, x="TAXA_ABSTENCOES", y="NM_MUNICIPIO", color="tomato", label="Abstenções", ax=ax)
# Add a legend and informative axis label
ax.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
ax.set(xlabel="Proporção")
ax.grid(visible=True, axis="x")
sns.despine(left=True, bottom=True)
plt.show()

# 4.2 Municípios com maior e menor taxa de comparecimento
max_row = taxa_comp_mun.orderBy(col("TAXA_COMPARECIMENTO").desc()).head()
min_row = taxa_comp_mun.orderBy(col("TAXA_COMPARECIMENTO")).head()
min_max = spark.createDataFrame([max_row, min_row])
min_max.show()

# 4.3 Quantidade de votos recebidos por partido
votos_partido = (data_cleaned
                 .groupBy("NR_PARTIDO", "SG_PARTIDO", "NM_PARTIDO")
                 .agg(
                     sum("QT_VOTOS").alias("QT_VOTOS")
                    )
                 .filter(col("SG_PARTIDO")!="#NULO#")
                 .orderBy(col("QT_VOTOS").desc())
                )
votos_partido.show(truncate=False)

data = votos_partido.toPandas()

# Configurações do gráfico
plt.figure(figsize=(10, 6))
sns.barplot(y='SG_PARTIDO', data=data, x='QT_VOTOS')

# Configurações dos eixos e título
plt.xlabel('Partido')
plt.ylabel('Contagem de Votos')
plt.title('Contagem de Votos por Partido')

# Exibir o gráfico
plt.show()

# 4.4 - Desempenho dos partidos por município
votos_partido_mun = (data_cleaned
                 .groupBy("NM_MUNICIPIO", "SG_PARTIDO", "NM_PARTIDO")
                 .agg(
                     sum("QT_VOTOS").alias("QT_VOTOS")
                    )
                 .filter(col("SG_PARTIDO")!="#NULO#")
                 .orderBy(col("NM_MUNICIPIO"), col("QT_VOTOS").desc())
                )

# Total de votos por município
total_votes_mun = votos_partido_mun.groupBy("NM_MUNICIPIO").agg(sum("QT_VOTOS").alias("TOTAL_VOTOS"))

# Juntando informação de total de votos ao dataframe
votos_partido_mun = votos_partido_mun.join(total_votes_mun, "NM_MUNICIPIO")

# Calculando percentual
votos_partido_mun = (votos_partido_mun
                        .withColumn("PERCENTUAL_VOTOS", round(col("QT_VOTOS") / col("TOTAL_VOTOS") * 100, 2))
                        .orderBy(col("PERCENTUAL_VOTOS").desc())
                    )
votos_partido_mun.show(35, truncate=False)

# 4.5 - Candidatos mais votados por cargo
votos_cand = (data_cleaned
                 .filter(col("CD_TIPO_VOTAVEL")==1)
                 .groupBy("DS_CARGO_PERGUNTA", "NM_VOTAVEL")
                 .agg(
                     sum("QT_VOTOS").alias("QT_VOTOS")
                    )
                 .orderBy(col("DS_CARGO_PERGUNTA").desc(), col("QT_VOTOS").desc())
                )

# Create a window specification partitioned by 'DS_CARGO_PERGUNTA' and ordered by 'QT_VOTOS' in descending order
window_spec = Window.partitionBy("DS_CARGO_PERGUNTA").orderBy(col("QT_VOTOS").desc())

# Filter the DataFrame to select only the rows with row number = 1 within each 'DS_CARGO_PERGUNTA' group
votos_cand = (votos_cand
                    .withColumn("row_number", F.row_number().over(window_spec)) \
                    .filter(col("row_number") == 1) \
                    .drop("row_number")
                    .orderBy(col("QT_VOTOS").desc())
             )

votos_cand.show(truncate=False)

# 4.6 - Regiões com maior concentração de votos em determinados partidos ou candidatos

votos_partido_mun = (data_cleaned
                 .groupBy("NM_MUNICIPIO", "SG_PARTIDO", "NM_PARTIDO", "DS_CARGO_PERGUNTA", "NM_VOTAVEL")
                 .agg(
                     sum("QT_VOTOS").alias("QT_VOTOS")
                    )
                 .filter(col("SG_PARTIDO")!="#NULO#")
                 .orderBy(col("NM_MUNICIPIO"), col("QT_VOTOS").desc())
                )

# Split the string and create a list of tuples
reg_string = """Agreste Potiguar | Agreste Potiguar | Bom Jesus; Brejinho; Ielmo Marinho; Januário Cicco; Jundiá; Lagoa d'Anta; Lagoa de Pedras; Lagoa Salgada; Monte Alegre; Nova Cruz; Passa e Fica; Passagem; Presidente Juscelino; Riachuelo; Santa Maria; Santo Antônio; São Paulo do Potengi; São Pedro; Senador Elói de Souza; Serrinha; Várzea; Vera Cruz; Boa Saúde; Serra Caiada
Agreste Potiguar | Baixa Verde | Bento Fernandes; Jandaíra; João Câmara; Parazinho; Poço Branco
Agreste Potiguar | Borborema Potiguar | Barcelona; Campo Redondo; Coronel Ezequiel; Jaçanã; Japi; Lagoa de Velhos; Lajes Pintadas; Monte das Gameleiras; Ruy Barbosa; Santa Cruz; São Bento do Trairí; São José do Campestre; São Tomé; Serra de São Bento; Sítio Novo; Tangará
Central Potiguar | Angicos | Afonso Bezerra; Angicos; Caiçara do Rio do Vento; Fernando Pedroza; Jardim de Angicos; Lajes; Pedra Preta; Pedro Avelino
Central Potiguar | Macau | Caiçara do Norte; Galinhos; Guamaré; Macau; São Bento do Norte
Central Potiguar | Seridó Ocidental Potiguar | Caicó; Ipueira; Jardim de Piranhas; São Fernando; São João do Sabugi; Serra Negra do Norte; Timbaúba dos Batistas
Central Potiguar | Seridó Oriental Potiguar | Acari; Carnaúba dos Dantas; Cruzeta; Currais Novos; Equador; Jardim do Seridó; Ouro Branco; Parelhas; Santana do Seridó; São José do Seridó
Central Potiguar | Serra de Santana | Bodó; Cerro Corá; Florânia; Lagoa Nova; Santana do Matos; São Vicente; Tenente Laurentino Cruz
Leste Potiguar | Litoral Nordeste Potiguar | Maxaranguape; Pedra Grande; Pureza; Rio do Fogo; São Miguel de Touros; Taipu; Touros; Arez; São miguel do gostoso
Leste Potiguar | Litoral Sul Potiguar | Arês; Baía Formosa; Canguaretama; Espírito Santo; Goianinha; Montanhas; Pedro Velho; Senador Georgino Avelino; Tibau do Sul; Vila Flor
Leste Potiguar | Macaíba | Ceará-Mirim; Macaíba; Nísia Floresta; São Gonçalo do Amarante; São José de Mipibu
Leste Potiguar | Natal | Extremoz; Natal; Parnamirim
Oeste Potiguar | Chapada do Apodi | Apodi; Caraúbas; Felipe Guerra; Governador Dix-Sept Rosado
Oeste Potiguar | Médio Oeste Potiguar | Augusto Severo; Janduís; Messias Targino; Paraú; Triunfo Potiguar; Upanema; Campo Grande
Oeste Potiguar | Mossoró | Areia Branca; Baraúna; Grossos; Mossoró; Serra do Mel; Tibau
Oeste Potiguar | Pau dos Ferros | Alexandria; Francisco Dantas; Itaú; José da Penha; Marcelino Vieira; Paraná; Pau dos Ferros; Pilões; Portalegre; Rafael Fernandes; Riacho da Cruz; Rodolfo Fernandes; São Francisco do Oeste; Severiano Melo; Taboleiro Grande; Tenente Ananias; Viçosa
Oeste Potiguar | Serra de São Miguel | Água Nova; Coronel João Pessoa; Doutor Severiano; Encanto; Luís Gomes; Major Sales; Riacho de Santana; São Miguel; Venha-Ver
Oeste Potiguar | Umarizal | Almino Afonso; Antônio Martins; Frutuoso Gomes; João Dias; Lucrécia; Martins; Olho d'Água do Borges; Patu; Rafael Godeiro; Serrinha dos Pintos; Umarizal
Oeste Potiguar | Vale do Açu | Assú; Alto do Rodrigues; Carnaubais; Ipanguaçu; Itajá; Jucurutu; Pendências; Porto do Mangue; São Rafael"""

reg_list = [item.split("|") for item in reg_string.split("\n")]

# Create an empty DataFrame with a defined schema
df_regions = spark.createDataFrame([], schema="MACRO: string, MICRO: string, municipio: string")

# Append rows to the DataFrame
for item in reg_list:
    macro = item[0].strip().upper()
    micro = item[1].strip().upper()
    cities = item[2].split(";")
    
    for city in cities:
        df_regions = df_regions.union(
            spark.createDataFrame([(macro, micro, city.strip().upper())], schema="MACRO: string, MICRO: string, municipio: string")
        )
# Merge df_regions with data_df
data_reg_df = votos_partido_mun.join(df_regions, votos_partido_mun["NM_MUNICIPIO"] == df_regions["municipio"], "left")
data_reg_df = data_reg_df.select(votos_partido_mun.columns + df_regions.columns[:2])
data_reg_df.show(truncate=False)

## 4.6.1 Macro
macro = (data_reg_df
            .groupBy("DS_CARGO_PERGUNTA", "NM_VOTAVEL", "MACRO")
             .agg(
                 sum("QT_VOTOS").alias("QT_VOTOS")
             )
             .orderBy(col("QT_VOTOS").desc())
        )
macro.show(truncate=False)

## 4.6.1 Micro
micro = (data_reg_df
            .groupBy("DS_CARGO_PERGUNTA", "NM_VOTAVEL", "MICRO")
             .agg(
                 sum("QT_VOTOS").alias("QT_VOTOS")
             )
             .orderBy(col("QT_VOTOS").desc())
        )
micro.show(truncate=False)

# 4.7 - Proporção de votos válidos, brancos e nulos
votos_tipo = (data_cleaned
                 .groupBy("DS_CARGO_PERGUNTA", "DS_TIPO_VOTAVEL")
                 .agg(
                     sum("QT_VOTOS").alias("QT_VOTOS"),
                    )
                 #.withColumn("TAXA_COMPARECIMENTO", round((col("QT_COMPARECIMENTO") / col("QT_APTOS") * 100), 2))
                 .orderBy("DS_CARGO_PERGUNTA", "DS_TIPO_VOTAVEL")
                )
votos_tipo.show(truncate=False)

# Total de votos por cargo
votos_nulos = votos_tipo.groupBy("DS_CARGO_PERGUNTA").agg(sum("QT_VOTOS").alias("TOTAL_VOTOS"))

# Juntando informação de total de votos ao dataframe
votos_nulos = votos_tipo.join(votos_nulos, "DS_CARGO_PERGUNTA")

# Calculando percentual
votos_nulos = (votos_nulos
                        .withColumn("PERCENTUAL_VOTOS", round(col("QT_VOTOS") / col("TOTAL_VOTOS") * 100, 2))
                        .orderBy(col("DS_CARGO_PERGUNTA"), col("PERCENTUAL_VOTOS").desc())
                    )
votos_nulos.show(truncate=False)