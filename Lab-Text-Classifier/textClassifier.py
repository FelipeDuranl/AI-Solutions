from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import numpy as np


# Dataset
textos = [
    # Tecnologia
    "Novo processador da Intel bate recordes de desempenho",
    "Empresa lança smartphone dobrável com tela de 7 polegadas",
    "Atualização do sistema operacional corrige falhas críticas",
    "Inteligência artificial cria músicas originais em segundos",
    "Startup brasileira desenvolve bateria que carrega em 5 minutos",
    "Mercado de chips semicondutores cresce 30% no ano",
    "Novo modelo de linguagem supera humanos em testes de leitura",
    "Empresa de computação quântica anuncia avanço histórico",
    "Wearable monitora saúde em tempo real com precisão médica",
    "Big tech investe bilhões em data centers no Brasil",
    # Esportes
    "Seleção brasileira vence por 3 a 1 nas eliminatórias",
    "Tenista conquista seu quarto título em Grand Slam consecutivo",
    "Clube anuncia contratação de atacante por 80 milhões",
    "Atleta quebra recorde mundial nos 100 metros rasos",
    "Fórmula 1 define calendário com 24 etapas para a temporada",
    "NBA: jogador marca 60 pontos em partida histórica",
    "Vôlei feminino do Brasil conquista ouro no mundial",
    "Maratona de São Paulo bate recorde de participantes",
    "Técnico é demitido após sequência de derrotas no campeonato",
    "Copa do Mundo feminina define grupos da fase de grupos",
    # Política
    "Presidente sanciona nova lei de reforma tributária",
    "Oposição critica cortes no orçamento da educação",
    "Diplomatas se reúnem para negociar acordo de paz no conflito",
    "Congresso aprova projeto de lei sobre privacidade de dados",
    "Partido anuncia candidato para as próximas eleições municipais",
    "Governo federal lança programa de habitação popular",
    "Ministro defende reforma previdenciária em audiência pública",
    "Senado vota proposta de emenda constitucional amanhã",
    "Cúpula do G20 discute crise climática e economia global",
    "Acordo bilateral entre países facilita comércio na região",
]

categorias = (["tecnologia"] * 10) + (["esportes"] * 10) + (["política"] * 10)

# Stopwords
stopwords_pt = [
    "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "em", "com", "por", "para", "um", "uma", "o", "a", "os",
    "as", "e", "que", "se", "ao", "após", "seu", "sua"
]

# Pipeline vectorizer e o classificador
pipeline = make_pipeline(
    TfidfVectorizer(stop_words=stopwords_pt, ngram_range=(1, 2)),
    LinearSVC()
)

# Testa em 5 splits diferentes e tira a média
scores = cross_val_score(pipeline, textos, categorias, cv=5)

print(f"Acurácias por fold: {scores.round(2)}")
print(f"Acurácia média:     {scores.mean():.2f}")
print(f"Desvio padrão:      {scores.std():.2f}")