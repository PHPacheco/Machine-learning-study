from a import QuestaoA
from b import QuestaoB
from c import QuestaoC
from d import QuestaoD
from e import QuestaoE
from f import QuestaoF
from g import QuestaoG
from h import QuestaoH
from i import QuestaoI
from j import QuestaoJ
from m import QuestaoM
from k import QuestaoK
from l import QuestaoL

def main():

    QuestaoA().executar()
    QuestaoB().executar(exibir=False)
    QuestaoC().executar(exibir=False)
    QuestaoD().executar(exibir=False)
    QuestaoE().executar(exibir=False)
    QuestaoF().executar(exibir=True) 
    QuestaoG().executar(exibir=True) 
    QuestaoJ().executar(exibir=True) 
    x_tr, x_te, y_tr, y_te = QuestaoH().executar()
    QuestaoI().executar(exibir=True)
    QuestaoM().executar(exibir=True)
    resultados = QuestaoK().executar()
    QuestaoL().executar(resultados)
    
if __name__ == "__main__":
    main()
