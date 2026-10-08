# Segurança de artefatos pickle

`pickle` não é um formato de dados seguro. A desserialização pode executar
código arbitrário, portanto um arquivo baixado, recebido por mensagem ou
modificado por terceiros não deve ser aberto.

O único pickle preservado é um artefato acadêmico sem fonte tabular disponível:
`classificadores-de-credito/dados/credit.pkl`. Antes de utilizá-lo:

1. confirme que o arquivo veio deste histórico Git privado;
2. revise o commit e a integridade do clone;
3. use um ambiente isolado e sem credenciais;
4. prefira migrar o conteúdo para CSV/Parquet assim que a fonte bruta for
   recuperada.

O loader bloqueia o uso acidental:

```python
from ml_studies.loaders import load_trusted_legacy_pickle

data = load_trusted_legacy_pickle(path, trusted=True)
```

`trusted=True` é uma confirmação explícita, não uma verificação de segurança.
