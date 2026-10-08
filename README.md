# Aprendizado de Máquina — estudos acadêmicos

Coleção organizada de trabalhos desenvolvidos na disciplina de Aprendizado de
Máquina da FURB. O repositório conserva o histórico e a autoria do trabalho em
grupo, ao mesmo tempo em que oferece um pacote Python reutilizável, testes e
exemplos com protocolos de avaliação mais seguros.

> **Escopo:** este acervo é material didático. Resultados históricos não devem
> ser interpretados como benchmarks atuais sem uma nova execução.

## Conteúdo

| Módulo | Assunto | Algoritmos | Dados | Situação dos resultados |
| --- | --- | --- | --- | --- |
| [`regressoes`](regressoes/) | regressão simples, múltipla e polinomial | mínimos quadrados, correlação de Pearson | TXT, CSV e MAT incluídos | três notebooks preservados e funções centrais testadas |
| [`classificacao-knn`](classificacao-knn/) | classificação baseada em vizinhos | kNN, `MinMaxScaler`, `GridSearchCV` | quatro grupos MAT incluídos | métricas geradas pelo pipeline, sem seleção de `k` no teste |
| [`classificadores-de-credito`](classificadores-de-credito/) | classificação tabular | Naive Bayes, árvore, floresta, regressão logística e SVM | risco de crédito em CSV e um artefato legado | conclusões e métricas geradas em execução |
| [`redes-neurais`](redes-neurais/) | redes densas e convolucionais | TensorFlow/Keras e PyTorch | Breast Cancer Wisconsin e Fashion-MNIST | três notebooks didáticos; execução completa requer extras pesados |

O trabalho final de identificação facial não faz parte desta coleção. Ele foi
separado em um [repositório próprio](https://github.com/PHPacheco/reconhecimento-facial-lfw),
que poderá ser renomeado posteriormente sem alterar a organização deste acervo.

O estado de cada notebook e seus bloqueios conhecidos estão em
[`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md).
As métricas obtidas nesta validação estão em
[`docs/RESULTADOS_REPRODUZIDOS.md`](docs/RESULTADOS_REPRODUZIDOS.md).

## Instalação

É necessário Python 3.11 ou 3.12. Em um clone limpo:

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev,notebooks]"

# Linux/macOS
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,notebooks]"
```

TensorFlow e PyTorch são opcionais e deliberadamente separados:

```bash
python -m pip install -e ".[tensorflow]"
python -m pip install -e ".[pytorch]"
```

Nenhum notebook instala dependências durante a execução.

## Execução

Os exemplos aplicam o pré-processamento aprendido somente no treino. No kNN,
os hiperparâmetros são selecionados por validação cruzada e o teste é usado uma
única vez:

```bash
python regressoes/exemplo.py
python classificacao-knn/exemplo.py
python classificadores-de-credito/exemplo.py
python tools/execute_smoke_notebook.py classificadores-de-credito/classificadores_credito.ipynb
```

Para abrir o material canônico:

```bash
jupyter lab
```

Qualidade e testes:

```bash
ruff check src tests tools regressoes/exemplo.py classificacao-knn/exemplo.py classificadores-de-credito/exemplo.py
ruff format --check src tests tools regressoes/exemplo.py classificacao-knn/exemplo.py classificadores-de-credito/exemplo.py
python -m pytest
python tools/check_notebooks.py
python tools/execute_smoke_notebook.py
```

## Metodologia e limitações

- Sementes são fixadas nos fluxos novos quando há aleatoriedade.
- kNN usa `Pipeline` e `GridSearchCV`; o pré-processamento é aprendido apenas
  no treino de cada fold.
- Macro-F1 e balanced accuracy acompanham a acurácia nos classificadores
  reorganizados, reduzindo o risco de conclusões enganosas em classes
  desbalanceadas.
- O código legado foi mantido quando tem valor didático, mas nem todo resultado
  antigo foi reproduzido. Limitações específicas constam no inventário de
  notebooks.
- Datasets pequenos e acadêmicos não representam, por si só, desempenho em
  produção nem generalização para outras populações.

## Artefatos `pickle`

Os pickles de risco de crédito e a cópia duplicada de `credit.pkl` foram
removidos. Mantém-se somente `classificadores-de-credito/dados/credit.pkl`, pois os dados
brutos correspondentes não estavam no histórico. O loader exige
`trusted=True` e emite um aviso: **pickle pode executar código arbitrário e
nunca deve ser carregado de origem não confiável**. Consulte
[`docs/PICKLE_SECURITY.md`](docs/PICKLE_SECURITY.md).

## Autoria e contribuições

O histórico registra contribuições de **Pedro Albuquerque**, **Julia Welter**,
**Tiago Roberto de Melo** (também registrado como Tiago) e **Pedro Pacheco**.
Os commits de Pedro Pacheco identificados no histórico abrangem exercícios de
regressão, o estudo kNN D2, partes de regressão logística/SVM e o trabalho de
CNN. A contribuição específica de Pedro Pacheco ao experimento facial não pôde
ser inferida: os dois commits do notebook estão atribuídos a Pedro Albuquerque
e essa informação precisa ser confirmada pelos autores antes de ser descrita.

O repositório não apresenta o trabalho coletivo como autoria individual. Use
`git log --follow -- <arquivo>` para consultar a proveniência de um arquivo.

## Licença e publicação

Não há licença open source atribuída ao código coletivo. A autorização dos
coautores para publicação separada e licenciamento ainda está pendente; por
isso, esta reorganização deve permanecer privada até confirmação. Datasets e
bibliotecas mantêm seus próprios termos. Em especial, arquivos do LFW não são
redistribuídos aqui.

## Estrutura

```text
src/ml_studies/       funções reutilizáveis de regressão, kNN e loaders
tests/                testes unitários leves
docs/                 decisões, segurança e estado dos notebooks
regressoes/            três projetos de regressão e seus dados
classificacao-knn/     notebook, exemplo e quatro datasets de kNN
classificadores-de-credito/  notebook, exemplo e dados acadêmicos
redes-neurais/         notebooks TensorFlow/Keras, PyTorch e CNN
.github/workflows/    CI leve, sem treinamento de redes
```
