from a import QuestaoA
from b import QuestaoB
from c import QuestaoC
from d import QuestaoD
from e import QuestaoE
from f import QuestaoF
from k import QuestaoK
from l import QuestaoL


def main():
    QuestaoA().executar()
    QuestaoB().executar(exibir=False)
    QuestaoC().executar(exibir=False)
    QuestaoD().executar(exibir=False)
    QuestaoE().executar(exibir=False)
    QuestaoF().executar(exibir=True)
    resultados = QuestaoK().executar()
    QuestaoL().executar(resultados)


if __name__ == "__main__":
    main()
