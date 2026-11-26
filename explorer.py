# import json

# import numpy as np

# from llm_sdk import Small_LLM_Model

# # ---------- Setup ----------
# model = Small_LLM_Model()

# with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
#     vocab = json.load(f)

# id_to_text = {}
# for text in vocab:
#     id_to_text[vocab[text]] = text

# function_names = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]


# # ---------- Constraint rules ----------
# def is_prefix_of_any_name(text, names):
#     for name in names:
#         if name.startswith(text):
#             return True
#     return False


# def allowed_tokens(generated_text, names, id_to_text):
#     allowed = []
#     for token_id in id_to_text:
#         candidate = generated_text + id_to_text[token_id]
#         if is_prefix_of_any_name(candidate, names):
#             allowed.append(token_id)
#     return allowed


# def pick_best_allowed(logits, allowed):
#     scores = np.full(len(logits), -np.inf)
#     for token_id in allowed:
#         scores[token_id] = logits[token_id]
#     return int(np.argmax(scores))


# # ---------- Prompt ----------
# request = "Greet shrek"
# prompt = (
#     "Available functions:\n"
#     "fn_add_numbers: Add two numbers together and return their sum.\n"
#     "fn_greet: Generate a greeting message for a person by name.\n"
#     "fn_reverse_string: Reverse a string and return the reversed result.\n"
#     "\n"
#     "Request: " + request + "\n"
#     "Function name:\n"
# )

# # ---------- Generation loop ----------
# ids = model.encode(prompt).tolist()[0]
# generated_text = ""
# step = 0

# while generated_text not in function_names and step < 20:
#     logits = model.get_logits_from_input_ids(ids)
#     allowed = allowed_tokens(generated_text, function_names, id_to_text)
#     chosen = pick_best_allowed(logits, allowed)
#     ids.append(chosen)
#     generated_text = generated_text + id_to_text[chosen]
#     step = step + 1

# print("Chosen function:", generated_text)




import json
import numpy as np

from llm_sdk import Small_LLM_Model

# 1. Carregar o modelo (a primeira vez demora: faz o download)
model = Small_LLM_Model()

# 2. Texto -> IDs
# frase = "What is the sum of 2 and 3? "
# ids = model.encode(frase).tolist()[0]
# print("IDs:", ids)
# print("Número de tokens:", len(ids))

# 3. Abrir o vocabulário (formato: texto -> id)
caminho = model.get_path_to_vocab_file()
with open(caminho, "r", encoding="utf-8") as f:
    vocab = json.load(f)
print("Tamanho do vocabulário:", len(vocab))

# 4. Inverter: id -> texto
id_para_texto = {}
for texto in vocab:
    numero = vocab[texto]
    id_para_texto[numero] = texto

# 5. Mostrar cada token ao lado do seu ID
# for i in ids:
#     print(i, "->", repr(id_para_texto[i]))

# 6. Pedir os logits ao modelo
# logits = model.get_logits_from_input_ids(ids)
# print("Tamanho dos logits:", len(logits))
# print("Diferença: ", len(logits) - len(vocab), "\n")


# for passo in range(15):
#     logits = model.get_logits_from_input_ids(ids)
#     melhor_id = int(np.argmax(logits))
#     ids.append(melhor_id)
#     print(passo, repr(model.decode([melhor_id])))

nomes = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]


def e_inicio_de_algum_nome(texto, nomes):
    for nome in nomes:
        if nome.startswith(texto):
            return True
    return False


def tokens_permitidos(texto_gerado, nomes, id_para_texto):
    permitidos = []
    for token_id in id_para_texto:
        texto_token = id_para_texto[token_id]
        candidato = texto_gerado + texto_token
        if e_inicio_de_algum_nome(candidato, nomes):
            permitidos.append(token_id)
    return permitidos


# print("=== Primeiro passo (nada gerado) ===")
# p = tokens_permitidos("f", nomes, id_para_texto)
# print("Quantos:", len(p))
# for token_id in p:
#     print(token_id, repr(id_para_texto[token_id]))

# print("=== Depois de 'fn_' ===")
# p = tokens_permitidos("fn_", nomes, id_para_texto)
# print("Quantos:", len(p))
# for token_id in p:
#     print(token_id, repr(id_para_texto[token_id]))

pedido = "Write 'banana' backwards"

prompt = (
    "Available functions:\n"
    "fn_add_numbers: Add two numbers together and return their sum.\n"
    "fn_greet: Generate a greeting message for a person by name.\n"
    "fn_reverse_string: Reverse a string and return the reversed result.\n"
    "\n"
    "Request: " + pedido + "\n"
    "Function name:\n"
)


def escolher_permitido(logits, permitidos):
    notas = np.full(len(logits), -np.inf)
    for token_id in permitidos:
        notas[token_id] = logits[token_id]
    return int(np.argmax(notas))


ids = model.encode(prompt).tolist()[0]
texto_gerado = ""
passo = 0

print("=== ANTES DO CICLO ===")
print("len(ids):", len(ids))
print("últimos 5 ids:", ids[-5:])
print("texto_gerado:", repr(texto_gerado))
print()

while texto_gerado not in nomes and passo < 20:
    print("=== PASSO", passo, "===")
    print("texto_gerado (início do passo):", repr(texto_gerado))
    print("len(ids):", len(ids))

    # 1. notas do modelo
    logits = model.get_logits_from_input_ids(ids)
    favorito = int(np.argmax(logits))
    print("favorito SEM restrições:", favorito,
          repr(id_para_texto.get(favorito, "<especial>")),
          "nota:", round(logits[favorito], 2))

    # 2. permitidos
    permitidos = tokens_permitidos(texto_gerado, nomes, id_para_texto)

    print("permitidos (", len(permitidos), "):")
    for token_id in permitidos:
        print("   ", token_id, repr(id_para_texto[token_id]),
              "nota:", round(logits[token_id], 2))

    # 3. escolha
    escolhido = escolher_permitido(logits, permitidos)
    print("escolhido COM restrições:", escolhido,
          repr(id_para_texto[escolhido]))

    # 4. atualizar
    ids.append(escolhido)
    texto_gerado = texto_gerado + id_para_texto[escolhido]
    print("texto_gerado (fim do passo):", repr(texto_gerado))
    print("está nos nomes?", texto_gerado in nomes)
    print()
    passo = passo + 1

print("=== FIM ===")
print("Função escolhida:", texto_gerado)
print("passos usados:", passo)
