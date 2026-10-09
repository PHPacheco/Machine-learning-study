# Estudos de Aprendizado de Máquina

Este repositório reúne exercícios e trabalhos desenvolvidos durante a disciplina
de Aprendizado de Máquina da FURB. A ideia é manter os projetos em um só lugar,
com os notebooks e os dados usados nas atividades.

Os trabalhos foram feitos em grupo, e o histórico do Git conserva a autoria de
cada contribuição.

## Projetos

| Pasta | Conteúdo |
| --- | --- |
| [`regressoes`](regressoes/) | Regressão linear simples, regressão múltipla e regressão polinomial |
| [`classificacao-knn`](classificacao-knn/) | Classificação com kNN em quatro conjuntos de dados |
| [`classificadores-de-credito`](classificadores-de-credito/) | Comparação entre Naive Bayes, random forest, regressão logística e SVM |
| [`redes-neurais`](redes-neurais/) | Redes densas com TensorFlow e PyTorch, além de uma CNN para o Fashion-MNIST |

O trabalho final de identificação facial foi separado em
[outro repositório](https://github.com/PHPacheco/reconhecimento-facial-lfw).

## Como executar

Use Python 3.11 ou 3.12. Para preparar o ambiente:

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -r requirements.txt

# Linux e macOS
.venv/bin/python -m pip install -r requirements.txt
```

Abra o notebook que deseja estudar. Por exemplo:

```bash
jupyter lab regressoes/regressao_linear_simples.ipynb
jupyter lab regressoes/regressao_linear_multipla.ipynb
jupyter lab regressoes/regressao_polinomial.ipynb
jupyter lab classificacao-knn/classificacao_knn.ipynb
jupyter lab classificadores-de-credito/classificadores_credito.ipynb
jupyter lab redes-neurais/rede_neural_tensorflow.ipynb
jupyter lab redes-neurais/rede_neural_pytorch.ipynb
jupyter lab redes-neurais/rede_convolucional_fashion_mnist.ipynb
```

As dependências de TensorFlow e PyTorch são opcionais. Os comandos estão no
README da pasta [`redes-neurais`](redes-neurais/).

## Observações sobre os experimentos

Nos estudos de kNN e SVM, a escolha dos hiperparâmetros é feita apenas com os
dados de treino. O conjunto de teste fica reservado para a avaliação final.
Além da acurácia, as avaliações mostram macro-F1 e balanced accuracy.

Os resultados dependem da divisão dos dados, das sementes e das versões das
bibliotecas. Por isso, devem ser lidos como resultados dos exercícios, não como
referência de desempenho para uso em produção.

O arquivo `classificadores-de-credito/dados/credit.pkl` foi mantido porque a
fonte tabular equivalente não está disponível no histórico. Arquivos `pickle`
podem executar código durante a leitura. Não substitua esse arquivo por uma
cópia obtida de uma fonte desconhecida.

## Autoria

O histórico contém contribuições de:

- [Pedro Krubniki Albuquerque (@pkalbuquerque)](https://github.com/pkalbuquerque);
- [Julia Welter (@juliawel)](https://github.com/juliawel);
- [Tiago Roberto de Melo (@TiagoRMelo)](https://github.com/TiagoRMelo);
- [Pedro Pacheco (@PHPacheco)](https://github.com/PHPacheco).

As contribuições de Pedro Pacheco incluem exercícios de regressão, o estudo
kNN D2, partes dos classificadores de crédito e o trabalho de CNN. A autoria
detalhada de cada arquivo pode ser consultada com
`git log --follow -- <arquivo>`.

## Licença

O código coletivo não recebeu uma licença open source. A autorização dos
coautores para uma publicação separada ainda precisa ser confirmada. Até essa
confirmação, o repositório deve permanecer privado.
