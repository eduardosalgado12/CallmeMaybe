
from pydantic import BaseModel


class Pessoa(BaseModel):
    nome: str
    idade: int


p = Pessoa(nome="Eduardo", idade=30)
print(p)
print(p.nome)
print(p.idade)
