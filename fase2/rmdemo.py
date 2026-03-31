import matplotlib.pyplot as plt

from exercicio_a import executar as executar_a
from exercicio_b import executar as executar_b
from exercicio_c import executar as executar_c
from exercicio_d import executar as executar_d
from exercicio_e import executar as executar_e
from exercicio_f import executar as executar_f
from exercicio_g import executar as executar_g
from exercicio_h import executar as executar_h
from exercicio_i import executar as executar_i


def main():
    base = executar_a()
    print()
    executar_b(base)
    print()
    executar_c(base)
    print()
    executar_d(base)
    executar_e(base)
    executar_f(base)
    executar_g(base)
    print()
    executar_h(base)
    print()
    executar_i(base)
    plt.show()


if __name__ == "__main__":
    main()
