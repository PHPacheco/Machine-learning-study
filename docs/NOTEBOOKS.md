# Inventário dos notebooks canônicos

Todos os notebooks abaixo têm kernel Python 3, saídas limpas e nenhuma célula
de instalação. Limpar saídas evita publicar caminhos pessoais e impede que
métricas antigas pareçam ter sido reproduzidas nesta reorganização.

| Notebook | Status de validação | Observação |
| --- | --- | --- |
| `1_Regressão/fase1/fase1.ipynb` | estrutura validada | material legado; as funções centrais têm testes unitários |
| `1_Regressão/fase2/fase2.ipynb` | estrutura validada | dados CSV/MAT incluídos; imports corrigidos nos scripts |
| `1_Regressão/fase3/fase3.ipynb` | estrutura validada | execução visual não faz parte da CI headless |
| `2_kNN/knn_com_validacao.ipynb` | executado sequencialmente na CI | smoke test sintético, sem vazamento entre treino e teste |
| `3_NaiveBayes/classificadores_credito.ipynb` | executado localmente; estrutura validada na CI | versão canônica sem referências a pickles removidos; teste isolado até o final |
| `4_RNA_REDE_NEURAL/TRABALHO4_RNA_PARTE1_RESOLVIDO.ipynb` | bloqueio documentado | exige extra `tensorflow`; treinamento não é adequado à CI leve |
| `4_RNA_REDE_NEURAL/TRABALHO4_RNA_PARTE1_PYTORCH.ipynb` | bloqueio documentado | variante distinta, exige extra `pytorch` |
| `4_RNA_REDE_NEURAL/Trabalho_4_RNA_e_CNN_PARTE2.ipynb` | bloqueio documentado | versão canônica da CNN; exige TensorFlow e download do MNIST |

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
python tools/execute_smoke_notebook.py 3_NaiveBayes/classificadores_credito.ipynb
```
