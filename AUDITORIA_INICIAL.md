# Auditoria inicial — 2026-10-04

Este registro foi criado antes da reorganização do conteúdo. Ele descreve o
estado encontrado no commit `7aa5e89` da branch `main`.

## Git e publicação

- A árvore de trabalho estava limpa.
- O único remote era `origin`, apontando para
  `https://github.com/pkalbuquerque/AprendizadoDeMaquina`.
- A autenticação do GitHub CLI para `PHPacheco` estava inválida; portanto, não
  era possível criar o fork nem publicar os repositórios.
- Não havia `README.md`, `.gitignore`, manifesto de dependências, testes ou CI.
- Não havia arquivo de licença. A autorização dos coautores para relicenciar ou
  publicar separadamente ainda precisava ser confirmada.

## Inventário

- 132 caminhos versionados na listagem inicial.
- 20 notebooks Jupyter, incluindo versões alternativas e incompletas.
- 51 arquivos `.pyc`, todos dentro de diretórios `__pycache__`, versionados.
- 5 arquivos `pickle`; dois pares eram duplicatas byte a byte:
  - `Parte2/credit.pkl` e `Parte3/credit.pkl`;
  - `Parte2/risco_credito.pkl` e `Parte3/risco_credito.pkl`.
- O notebook de CNN existia em `4_RNA_REDE_NEURAL` e `Trabalho 4` com o mesmo
  código-fonte, mas saídas diferentes; ambos declaravam kernel Python 2.
- O notebook facial tinha 20 células, 9 células de código, 18 saídas e cerca de
  337 kB.

## Autoria observável no histórico

O histórico completo registra commits de Pedro Albuquerque, Julia Welter,
Tiago/Thiago Roberto de Melo e Pedro Pacheco. Para `Pedro Pacheco`, foram
localizados commits em regressão, kNN, regressão logística/SVM e CNN.

Os dois commits que introduziram e alteraram
`4_TrabalhoFinal/TrabalhoFinal_FaceRecognition.ipynb` (`1fe25e2` e `2c9a22b`)
estão atribuídos a Pedro Albuquerque. Uma eventual contribuição de Pedro
Pacheco ao experimento facial fora do Git precisa ser confirmada antes de ser
descrita como fato.

## Problemas técnicos e metodológicos

- Imports como `from Regressão...` não correspondem à pasta `1_Regressão` nem
  a um pacote Python importável.
- `3_NaiveBayes/Parte3/Demo.py` força o backend gráfico `TkAgg`, incompatível
  com execução headless.
- Há notebooks incompletos, versões concorrentes e metadados Python 2.
- Notebooks de redes neurais e o notebook facial contêm instalações com `pip`,
  logs de ambiente e/ou caminhos pessoais.
- Os artefatos `pickle` eram carregados diretamente, apesar de `pickle` poder
  executar código durante a desserialização de arquivos não confiáveis.
- O notebook facial usa um cache global (`encodings_cache.npz`) sem vincular o
  cache à versão do dataset ou aos parâmetros de seleção.
- O notebook facial mantém um teste separado, mas usa hiperparâmetros fixos sem
  validação explícita e compara apenas acurácia.
- Os resultados faciais de 99,42% (SVM) e 99,22% (kNN), 62 pessoas e 2.055
  imagens existem apenas como saídas previamente registradas; não foram
  reproduzidos nesta auditoria.
- A distribuição Kaggle declara LGPL-3.0 em seus metadados, mas o LFW agrega
  imagens coletadas da web e os direitos autorais das imagens permanecem com
  suas fontes originais. O dataset não deve ser redistribuído neste repositório.

## Restrições para a reorganização

- Preservar autores e commits do diretório facial por `git subtree split`.
- Não publicar como público nem adicionar licença ao código coletivo antes da
  autorização dos coautores.
- Não tratar resultados antigos como reproduzidos.
- Não baixar o LFW nem gerar embeddings completos na CI.
