# Inventário dos notebooks canônicos

Todos os notebooks abaixo têm kernel Python 3, saídas limpas e nenhuma célula
de instalação. Limpar saídas evita publicar caminhos pessoais e impede que
métricas antigas pareçam ter sido reproduzidas nesta reorganização.

| Notebook | Status de validação | Observação |
| --- | --- | --- |
| `regressoes/regressao_linear_simples.ipynb` | executado localmente em modo headless | material acadêmico; as funções centrais têm testes unitários |
| `regressoes/regressao_linear_multipla.ipynb` | executado localmente em modo headless | dados CSV/MAT incluídos; caminhos relativos à pasta do notebook |
| `regressoes/regressao_polinomial.ipynb` | executado localmente em modo headless | gráficos foram renderizados pelo backend não interativo |
| `classificacao-knn/classificacao_knn.ipynb` | executado sequencialmente na CI | smoke test sintético, sem vazamento entre treino e teste |
| `classificadores-de-credito/classificadores_credito.ipynb` | executado localmente; estrutura validada na CI | versão canônica; teste isolado até a avaliação final |
| `redes-neurais/rede_neural_tensorflow.ipynb` | bloqueio documentado | exige extra `tensorflow`; treinamento não é adequado à CI leve |
| `redes-neurais/rede_neural_pytorch.ipynb` | bloqueio documentado | variante distinta, exige extra `pytorch` |
| `redes-neurais/rede_convolucional_fashion_mnist.ipynb` | bloqueio documentado | CNN canônica; exige TensorFlow e download do Fashion-MNIST |

As alternativas byte a byte, versões incompletas e notebooks concorrentes
foram removidos na branch de reorganização; permanecem recuperáveis no
histórico Git. A versão facial foi separada por `git subtree split` para o
repositório `reconhecimento-facial-lfw`.

## Validação estrutural

```bash
python tools/check_notebooks.py
```

Esse comando verifica JSON, kernel Python 3, ausência de outputs, `%pip`,
`!pip`, caminhos pessoais conhecidos e metadados de Python 2.

O notebook leve também é executado célula por célula:

```bash
python tools/execute_smoke_notebook.py
```

O notebook canônico de classificadores pode ser reproduzido explicitamente;
ele emite o aviso de segurança antes de abrir o pickle legado versionado:

```bash
python tools/execute_smoke_notebook.py classificadores-de-credito/classificadores_credito.ipynb
```
